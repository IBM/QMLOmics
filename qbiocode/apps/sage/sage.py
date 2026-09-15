import logging
import os, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
import dill as pickle

# The complexity-column schema is owned by the evaluation layer, which produces it --
# not by this app, which only consumes it. qbiocode.visualization also consumes it.
from qbiocode.evaluation.dataset_evaluation import (
    SAMPLE_COUNT_COLUMN,
    detect_complexity_schema,
)

logger = logging.getLogger(__name__)


def _can_show() -> bool:
    """Whether ``plt.show()`` would display anything rather than warn and do nothing."""
    return mpl.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "cairo", "template")


#####


class QuantumSage():

    '''
    Sage class that will run an ML model over the input data frame which would be some set of defined data characeristics
    and performance metrics associated to the dataset the method use.  Right now it is focused on learning from just the data
    characteristics but it can eventual also include the model parameters as part of the input
    '''

    def __init__(self, data_input):
        '''
        This function initializes the Sage with the input data frame that contains the data characteristics and performance metrics
        '''

        # Detected rather than hardcoded: the complexity block QProfiler writes is
        # now pyMFE-backed, but the committed benchmark table predates that and
        # cannot be regenerated from this repository. Reading whichever schema the
        # caller supplies keeps both trainable -- see detect_complexity_schema.
        # The FRAME, not `data_input.columns`: only the rows reveal a table that
        # concatenates both schemas, and reading such a table as 'pymfe' trains the
        # sub-sages on a block of zeros. See detect_complexity_schema's Note.
        self._complexity_schema, self._columns_data_features = detect_complexity_schema(
            data_input
        )
        self._columns_metrics = ['accuracy', 'f1_score', 'auc']
        # The column recording how each model was parameterized is named for the
        # branch that produced it: model_evaluation.py writes 'BestParams_Tuned'
        # when grid_search is on and 'Model_Parameters' when it is off, never both.
        # 'BestParams_GridSearch' is the name the tuned branch used before Optuna
        # replaced the exhaustive grid, and is still what every ModelResults.csv
        # written before that change carries. Requiring both made
        # `data_input[self._columns_metadata]`
        # raise `KeyError: "['BestParams_GridSearch'] not in index"` on every
        # QProfiler run there has ever been -- so QuantumSage could not be
        # constructed from its own documented input in *either* configuration.
        # Whichever column is present is carried through; neither is trained on
        # (only 'model' and 'embeddings' are read below), so an input that
        # records no parameters at all is still usable.
        self._columns_metadata_required = [
            'Dataset', 'embeddings', 'datatype', 'model_embed_datatype', 'iteration', 'model',
        ]
        self._columns_parameters = [
            name for name in ('BestParams_Tuned', 'BestParams_GridSearch', 'Model_Parameters')
            if name in data_input.columns
        ]
        self._columns_metadata = self._columns_metadata_required + self._columns_parameters

        missing = [
            name for group in (self._columns_data_features, self._columns_metrics,
                               self._columns_metadata_required)
            for name in group if name not in data_input.columns
        ]
        if missing:
            raise ValueError(
                f"data_input is missing {len(missing)} required column(s): {missing}. "
                "QSage trains on a QProfiler results table (ModelResults.csv) with the "
                "metadata columns the QSage tutorial adds -- 'datatype', "
                "'model_embed_datatype' and 'iteration'. See "
                "tutorial/QSage/qsage.ipynb for the exact preparation step."
            )

        self._input_data_features_only = data_input[self._columns_data_features]
        self._input_data_metrics = data_input[self._columns_metrics]
        self._input_data_metadata = data_input[self._columns_metadata]

        self._available_models = list(set(self._input_data_metadata['model']))
        if 'none' in self._available_models:
            self._available_models.remove('none')
        self._available_models.sort()
        self._available_embeddings = list(set(self._input_data_metadata['embeddings']))
        self._available_embeddings.sort()
        # sorted(), not `= self._columns_metrics` followed by .sort(): that bound
        # both names to the same list, so sorting the public one silently
        # reordered the column list used to slice the input frame.
        self._available_metrics = sorted(self._columns_metrics)

        self._results_subsages = {}

        self.set_seed()

    # TODO: trained sage should predict over every metric so that the user can decide what they want predicted
    def predict(self, input_data, metric = 'f1_score'):
        '''
        Rank every model by its predicted score on one dataset.

        Pass the dataset-complexity columns named by ``_columns_data_features`` --
        the ``SLGH`` feature that training derives is recomputed here, so a value
        passed in for it is ignored rather than trusted. Rows are sorted by ``metric * r2`` descending: a model
        whose surrogate fits poorly cannot reach the top on a confident-looking
        point prediction alone, so read the ``r2`` column alongside the score.

        An unknown metric, a missing feature column, an untrained sage, or an input
        that is not exactly one row each raise -- earlier versions returned ``None``
        for an unknown metric and ranked on the first row of a multi-row input.

        Args:
            input_data (pd.DataFrame): Exactly one row, carrying at least the
                columns in ``_columns_data_features``. Extra columns are ignored.
            metric (str): One of the trained metrics (``f1_score``, ``auc``,
                ``accuracy``).

        Returns:
            predictions_df (pd.DataFrame): One row per model, with columns
                ``model``, ``<metric>``, ``r2`` and ``<metric>*r2``, ranked by the
                last of those.
        '''

        if not self._results_subsages:
            raise RuntimeError(
                "No sub-sages have been trained yet, so there is nothing to predict "
                "with. Call train_sub_sages() first, or load a QuantumSage that was "
                "trained and pickled earlier."
            )
        if metric not in self._results_subsages:
            raise ValueError(
                f"No sub-sage was trained for metric {metric!r}. Trained metrics are "
                f"{sorted(self._results_subsages)}."
            )

        # One row in, one ranking out. The returned frame is one row per *model*, so
        # a multi-row input has nowhere to go -- and the loop below took
        # `.predict(...)[0]`, quietly ranking on the first row and discarding the
        # rest. That is easy to hit by accident: complexity features are measured on
        # the *embedded* data, so a single dataset contributes a distinct feature row
        # per (embedding, iteration), and the obvious
        # `results_df[sage._columns_data_features].drop_duplicates()` yields several.
        # The old behaviour returned a confident-looking ranking for whichever row
        # happened to sort first, labelled as the dataset's.
        if len(input_data) != 1:
            raise ValueError(
                f"input_data must hold exactly one row of dataset-complexity features; "
                f"got {len(input_data)}. Features are measured on the embedded data, so "
                "one dataset has a separate row per (embedding, iteration) -- select the "
                "one you want to predict for, e.g. "
                "held_out_df[held_out_df['embeddings'] == 'pca'][sage._columns_data_features]"
                ".iloc[[0]], or call predict() once per row."
            )

        missing = [c for c in self._columns_data_features if c not in input_data.columns]
        if missing:
            raise ValueError(
                f"input_data is missing {len(missing)} of the {len(self._columns_data_features)} "
                f"dataset-complexity features the sub-sages were trained on: {missing}. "
                "Pass the feature columns of a QProfiler results table, e.g. "
                "results_df[sage._columns_data_features]."
            )

        # Reproduce the training-time feature derivation. train_sub_sages() appends
        # SLGH (Scaled Latent Geometric Hardness) to X *after* splitting, so every
        # fitted sub-sage expects 24 columns while _columns_data_features names only
        # the 23 it was derived from. Forwarding the caller's frame unchanged made
        # predict() raise sklearn's "Feature names seen at fit time, yet now
        # missing: - SLGH" for the exact input the docstring asks for -- so no
        # caller passing the documented columns could ever get a prediction.
        # Recomputed rather than accepted from the caller: SLGH is a row-wise
        # function of columns that are already present, so deriving it here cannot
        # disagree with training, whereas a value passed in could.
        input_data = calculate_SLGH(input_data[self._columns_data_features])

        predictions = []
        for model in self._available_models:
            pred = self._results_subsages[metric][model]['fit_model'].predict(input_data)[0]
            r2 = self._results_subsages[metric][model]['r2']
            predictions.append([model, pred, r2])
        predictions_df = pd.DataFrame( predictions, columns = ['model',metric,'r2'] )
        predictions_df[metric+'*r2'] = predictions_df[metric] * predictions_df['r2']
        predictions_df = predictions_df.sort_values(metric+'*r2', ascending=False)
        return predictions_df


    def train_sub_sages(self, test_size=0.2, sage_type='random_forest', n_iter=None, cv=5):
        """
        Train sub-sage predictors for each ML model and performance metric.
        
        This function trains regression models (Sage) that learn to predict
        model performance based on data complexity features. A separate sub-sage
        is trained for each combination of ML model and performance metric.
        
        Parameters
        ----------
        test_size : float, optional
            Proportion of data to use for testing (0.0 to 1.0). Default is 0.2.
        sage_type : str, optional
            Type of regressor to use as Sage. Must be one of:
            
            - 'random_forest': Random Forest with hyperparameter tuning (default)
            - 'mlp': Multi-Layer Perceptron with grid search
            - 'xgboost_optuna': XGBoost with Optuna optimization (state-of-the-art)
            - 'catboost_optuna': CatBoost with Optuna optimization
            
            Only ONE sage type can be selected per training run.
            
        n_iter : int, optional
            For Random Forest: number of hyperparameter search iterations (default: 50).
            For MLP: maximum number of training epochs (default: 1000).
            For XGBoost-Optuna and CatBoost-Optuna: number of Optuna trials (default: 100).
            If None, uses the default for the selected sage_type.
        cv : int, optional
            Number of cross-validation folds for hyperparameter evaluation.
            Default is 5. Used by all sage types.
        
        Returns
        -------
        None
            Results are stored in the internal ``_results_subsages`` dictionary with structure:
            
            .. code-block:: text
            
                {
                    'metric1': {
                        'model1': {
                            'fit_model': <trained model>,
                            'preds': <predictions on test set>,
                            'y_test': <true values>,
                            'params': <model parameters>,
                            'mae': <mean absolute error>,
                            'mse': <mean squared error>,
                            'rmse': <root mean squared error>,
                            'r2': <R² score>
                        },
                        ...
                    },
                    ...
                }
        
        Raises
        ------
        ValueError
            If sage_type is not one of the valid types.
        ImportError
            If sage_type is 'xgboost_optuna' but XGBoost or Optuna is not installed, or
            'catboost_optuna' but CatBoost is not installed.
        
        Notes
        -----
        The function iterates over all available metrics and models, training a
        separate predictor for each combination. Progress is printed during training.
        
        Only one sage type can be used per training run. If you want to compare
        different sage types, you need to train them separately and compare results.
        
        **Recommended Sage Type:**
        
        For best performance on continuous value prediction, use 'xgboost_optuna',
        which combines the power of gradient boosting with advanced Bayesian
        hyperparameter optimization. 'catboost_optuna' is the same idea with a
        different booster; on the small, wide tables QProfiler produces the two often
        disagree, so it is worth training both and comparing the reported R-squared
        rather than assuming either wins.
        
        Examples
        --------
        Train with Random Forest (default):
        
        >>> sage.train_sub_sages(test_size=0.2, sage_type='random_forest')
        
        Train with MLP:
        
        >>> sage.train_sub_sages(test_size=0.2, sage_type='mlp')
        
        Train with XGBoost-Optuna (state-of-the-art):
        
        >>> sage.train_sub_sages(test_size=0.2, sage_type='xgboost_optuna', n_iter=200)
        
        Train with CatBoost-Optuna:
        
        >>> sage.train_sub_sages(test_size=0.2, sage_type='catboost_optuna', n_iter=200)
        
        Train with custom hyperparameter search:
        
        >>> sage.train_sub_sages(sage_type='random_forest', n_iter=100, cv=10)
        
        See Also
        --------
        _sage_random_forest : Random Forest Sage implementation
        _sage_mlp : MLP Sage implementation
        _sage_xgboost_optuna : XGBoost with Optuna Sage implementation (state-of-the-art)
        _sage_catboost_optuna : CatBoost with Optuna Sage implementation
        predict : Make predictions using trained Sages
        """
        # Validate sage_type parameter
        valid_sage_types = ['random_forest', 'mlp', 'xgboost_optuna', 'catboost_optuna']
        if sage_type not in valid_sage_types:
            raise ValueError(
                f"Invalid sage_type '{sage_type}'. Must be one of {valid_sage_types}. "
                f"Only one sage type can be selected per training run."
            )
        
        for metric in self._available_metrics:
            print(f"Working on {metric}")

            self._results_subsages[metric] = {}
            for model in self._available_models:
                model_indices = self._input_data_metadata[ self._input_data_metadata['model'] == model ].index
                X = self._input_data_features_only.loc[ model_indices ]
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                y = self._input_data_metrics.loc[model_indices][metric].fillna(0).to_numpy()
                
                print(f"Working on {model}")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = self._seed)

                # Calculate SLGH (Scaled Latent Geometric Hardness)
                X_train = calculate_SLGH(X_train)
                X_test = calculate_SLGH(X_test)

                if sage_type == 'random_forest':
                    # Use default n_iter=50 for Random Forest if not specified
                    rf_n_iter = n_iter if n_iter is not None else 50
                    self._results_subsages[metric][model] = self._sage_random_forest(
                        X_train, X_test, y_train, y_test, n_iter=rf_n_iter, cv=cv
                    )
                elif sage_type == 'mlp':
                    # Use default n_iter=1000 for MLP if not specified
                    mlp_n_iter = n_iter if n_iter is not None else 1000
                    self._results_subsages[metric][model] = self._sage_mlp(
                        X_train, X_test, y_train, y_test, n_iter=mlp_n_iter, cv=cv
                    )
                elif sage_type == 'xgboost_optuna':
                    # Use default n_iter=100 for XGBoost-Optuna if not specified
                    xgb_n_iter = n_iter if n_iter is not None else 100
                    self._results_subsages[metric][model] = self._sage_xgboost_optuna(
                        X_train, X_test, y_train, y_test, n_iter=xgb_n_iter, cv=cv
                    )
                elif sage_type == 'catboost_optuna':
                    # Use default n_iter=100 for CatBoost-Optuna if not specified
                    cb_n_iter = n_iter if n_iter is not None else 100
                    self._results_subsages[metric][model] = self._sage_catboost_optuna(
                        X_train, X_test, y_train, y_test, n_iter=cb_n_iter, cv=cv
                    )

    def _sage_mlp(self, X_train, X_test, y_train, y_test, n_iter=1000, cv=5):
        """
        Train a Multi-Layer Perceptron (MLP) regressor as a Sage predictor with hyperparameter tuning.
        
        This function performs a grid search over MLP hyperparameters to find the best
        configuration, then makes predictions on the test set. The search uses cross-validation
        to evaluate different parameter combinations. Early stopping is used to prevent
        overfitting during training.
        
        The function is called internally by :meth:`train_sub_sages` and is not meant
        to be called directly by users. It is designed to work with preprocessed data
        that has been split into training and test sets.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (data complexity metrics).
        X_test : pd.DataFrame
            Test features (data complexity metrics).
        y_train : pd.Series
            Training labels (model performance values).
        y_test : pd.Series
            Test labels (model performance values).
        n_iter : int, optional
            Maximum number of training iterations (epochs) for the MLP. Default is 1000.
            Training will run for at most this many iterations, but may stop earlier
            if early stopping criteria are met (no improvement for 10 consecutive epochs).
            Higher values allow more training time but take longer to run.
            Lower values speed up training but may underfit if set too low.
        cv : int, optional
            Number of cross-validation folds for hyperparameter evaluation. Default is 5.
            Each parameter combination is evaluated using k-fold cross-validation to
            ensure robust performance estimates.
        
        Returns
        -------
        dict
            Dictionary containing:
            
            - 'fit_model' : GridSearchCV
                Trained MLP model with best parameters from grid search
            - 'preds' : np.ndarray
                Predictions on test set
            - 'y_test' : pd.Series
                True test labels
            - 'params' : dict
                Best hyperparameters found by grid search
            - 'mae' : float
                Mean Absolute Error on test set
            - 'mse' : float
                Mean Squared Error on test set
            - 'rmse' : float
                Root Mean Squared Error on test set
            - 'r2' : float
                R² score on test set
        
        Notes
        -----
        The hyperparameter search space includes:
        
        - hidden_layer_sizes: [(32, 10), (64, 32), (100,), (50, 25)]
        - activation: ['relu', 'tanh']
        - solver: ['adam', 'lbfgs']
        - alpha: [0.0001, 0.001, 0.01] (L2 regularization)
        - learning_rate: ['constant', 'adaptive']
        
        The MLP uses:
        
        - Early stopping with patience of 10 epochs (``n_iter_no_change=10``)
        - Adaptive learning rate starting at 0.001
        - Maximum iterations controlled by ``n_iter`` parameter
        - Automatic batch size
        - 10% validation split for early stopping
        
        **Interpretation of n_iter for MLP:**
        
        Unlike Random Forest where ``n_iter`` controls the number of hyperparameter
        search iterations, for MLP it controls the **maximum number of training epochs**.
        The actual training may stop earlier due to early stopping (if no improvement
        for 10 consecutive epochs). This allows you to control the training time while
        still benefiting from early stopping to prevent overfitting.
        
        See Also
        --------
        _sage_random_forest : Alternative Random Forest sub-sage
        train_sub_sages : Main training function that calls this method
        """
        from sklearn.model_selection import GridSearchCV

        # Define hyperparameter grid for MLP
        param_grid = {
            'hidden_layer_sizes': [(32, 10), (64, 32), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }

        # Initialize MLP with early stopping
        mlp = MLPRegressor(
            batch_size='auto',
            learning_rate_init=0.001,
            max_iter=n_iter,  # Use n_iter for maximum training iterations
            random_state=self._seed,
            n_iter_no_change=10,  # Fixed early stopping patience
            early_stopping=True,
            validation_fraction=0.1
        )

        # Initialize GridSearchCV with configurable cv parameter
        mlp_grid = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            scoring='r2'
        )
        
        # Train
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        mlp_grid.fit(X_train, y_train)
        preds = mlp_grid.predict(X_test)
        params = mlp_grid.best_params_

        # Evaluate on held out
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        result = {
            'fit_model': mlp_grid,
            'preds': preds,
            'y_test': y_test,
            'params': params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

        return result

    def _sage_xgboost_optuna(self, X_train, X_test, y_train, y_test, n_iter=100, cv=5):
        """
        Train an XGBoost regressor as a Sage predictor with advanced hyperparameter optimization using Optuna.
        
        This function uses Optuna, a state-of-the-art hyperparameter optimization framework, to find
        the best XGBoost configuration through Bayesian optimization. XGBoost is a gradient boosting
        algorithm known for its superior performance on tabular data and continuous value prediction.
        
        The function is called internally by :meth:`train_sub_sages` and is not meant to be called
        directly by users. It is designed to work with preprocessed data that has been split into
        training and test sets.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (data complexity metrics).
        X_test : pd.DataFrame
            Test features (data complexity metrics).
        y_train : pd.Series
            Training labels (model performance values).
        y_test : pd.Series
            Test labels (model performance values).
        n_iter : int, optional
            Number of Optuna trials for hyperparameter optimization. Default is 100.
            Higher values explore more parameter combinations but take longer.
            Each trial uses cross-validation to evaluate the parameter set.
        cv : int, optional
            Number of cross-validation folds for hyperparameter evaluation. Default is 5.
            Each parameter combination is evaluated using k-fold cross-validation to
            ensure robust performance estimates.
        
        Returns
        -------
        dict
            Dictionary containing:
            
            - 'fit_model' : xgb.XGBRegressor
                Trained XGBoost model with best parameters from Optuna optimization
            - 'preds' : np.ndarray
                Predictions on test set
            - 'y_test' : pd.Series
                True test labels
            - 'params' : dict
                Best hyperparameters found by Optuna
            - 'mae' : float
                Mean Absolute Error on test set
            - 'mse' : float
                Mean Squared Error on test set
            - 'rmse' : float
                Root Mean Squared Error on test set
            - 'r2' : float
                R² score on test set
            - 'study' : optuna.Study
                Complete Optuna study object with optimization history
        
        **Hyperparameter Search Space:**
        
        The optimization explores:
        
        - n_estimators: [50, 500] - number of boosting rounds
        - max_depth: [3, 10] - maximum tree depth
        - learning_rate: [0.001, 0.3] - step size shrinkage (log scale)
        - subsample: [0.6, 1.0] - fraction of samples per tree
        - colsample_bytree: [0.6, 1.0] - fraction of features per tree
        - min_child_weight: [1, 10] - minimum sum of instance weight in child
        - gamma: [0, 5] - minimum loss reduction for split
        - reg_alpha: [0, 1] - L1 regularization (log scale)
        - reg_lambda: [0, 10] - L2 regularization (log scale)
        
        **Installation Requirements:**
        
        To use this function, install the required packages:
        
        .. code-block:: bash
        
            pip install xgboost optuna
        
        **Performance Tips:**
        
        - Increase n_iter (e.g., 200-500) for better optimization on complex problems
        - Use higher cv (e.g., 10) for more robust evaluation with sufficient data
        - Monitor the Optuna study object to understand optimization progress
        - Consider using GPU acceleration for XGBoost on large datasets
        
        Examples
        --------
        Train with XGBoost-Optuna (via train_sub_sages):
        
        >>> sage.train_sub_sages(sage_type='xgboost_optuna', n_iter=200, cv=10)
        
        See Also
        --------
        _sage_mlp : MLP sub-sage with grid search
        _sage_random_forest : Random Forest sub-sage with randomized search
        train_sub_sages : Main training function that calls this method
        
        References
        ----------
        .. [1] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
               In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
               Discovery and Data Mining (pp. 785-794).
        .. [2] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna:
               A next-generation hyperparameter optimization framework. In Proceedings of
               the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data
               Mining (pp. 2623-2631).
        """

        from sklearn.model_selection import cross_val_score
        
        # Preprocess data
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Define objective function for Optuna
        def objective(trial):
            """Optuna objective function for hyperparameter optimization."""
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self._seed,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            # Create model with suggested parameters
            model = xgb.XGBRegressor(**params)
            
            # Evaluate with cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )
            
            return scores.mean()
        
        # Create Optuna study with pruning
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self._seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Optimize hyperparameters
        study.optimize(objective, n_trials=n_iter, show_progress_bar=False)
        
        # Get best parameters
        best_params = study.best_params
        best_params['random_state'] = self._seed
        best_params['n_jobs'] = -1
        best_params['verbosity'] = 0
        
        # Train final model with best parameters
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        # Make predictions
        preds = best_model.predict(X_test)
        
        # Evaluate on test set
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        result = {
            'fit_model': best_model,
            'preds': preds,
            'y_test': y_test,
            'params': best_params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'study': study  # Include study for analysis
        }
        
        return result


    def _sage_catboost_optuna(self, X_train, X_test, y_train, y_test, n_iter=100, cv=5):
        """
        Train a CatBoost regressor as a Sage predictor with hyperparameter optimization using Optuna.

        The CatBoost counterpart of :meth:`_sage_xgboost_optuna`, and deliberately the same
        shape: an Optuna TPE study over a continuous search space, scored by cross-validated
        R-squared, then a final refit at the best configuration. Having two independent
        gradient-boosting surrogates is useful precisely because they disagree -- CatBoost's
        symmetric trees and ordered boosting tend to behave differently from XGBoost's on the
        small, wide tables QProfiler produces.

        The function is called internally by :meth:`train_sub_sages` and is not meant to be called
        directly by users. It is designed to work with preprocessed data that has been split into
        training and test sets.

        Two CatBoost-specific details differ from the XGBoost version:

        * ``bootstrap_type`` is pinned to ``'Bernoulli'`` so that ``subsample`` is legal. Left
          unset, CatBoost chooses the scheme from the inferred loss and the Bayesian scheme it
          may pick rejects ``subsample`` outright.
        * ``allow_writing_files=False`` and ``verbose=False`` are passed to every model. Without
          the first, each of the ``n_iter * cv`` fits writes a ``catboost_info/`` directory into
          the current working directory; without the second, each logs a line per boosting
          iteration.
        * There is no ``min_data_in_leaf`` dimension, though it is the natural counterpart to
          the XGBoost version's ``min_child_weight``. CatBoost accepts it under any grow policy
          but honours it under only ``Depthwise`` and ``Lossguide``; at the default
          ``SymmetricTree`` the predictions at ``min_data_in_leaf=1`` and ``=60`` are identical.
          Searching it here would have spent trials distinguishing models that cannot differ.
          Add it together with a ``grow_policy`` dimension if it is wanted.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features -- the data-complexity measures.
        X_test : pd.DataFrame
            Test features.
        y_train : pd.Series
            Training targets -- the metric value being predicted.
        y_test : pd.Series
            Test targets.
        n_iter : int, optional
            Number of Optuna trials (default: 100).
        cv : int, optional
            Number of cross-validation folds used to score each trial (default: 5).

        Returns
        -------
        dict
            Keys match every other ``_sage_*`` method so downstream reporting is uniform:

            - 'fit_model' : CatBoostRegressor
                Trained model with the best parameters from the Optuna study.
            - 'preds' : np.ndarray
                Predictions on ``X_test``.
            - 'y_test' : pd.Series
                The held-out targets, passed through for convenience.
            - 'params' : dict
                Best parameters found, including the fixed settings above.
            - 'mae', 'mse', 'rmse', 'r2' : float
                Test-set error metrics.
            - 'study' : optuna.Study
                The completed study, for inspecting the search afterwards.

        See Also
        --------
        _sage_xgboost_optuna : The XGBoost equivalent, on which this is modelled.
        train_sub_sages : The method that calls this one.

        References
        ----------
        .. [1] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018).
               CatBoost: unbiased boosting with categorical features. In Advances in Neural
               Information Processing Systems 31 (pp. 6638-6648).
        .. [2] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna:
               A next-generation hyperparameter optimization framework. In Proceedings of
               the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data
               Mining (pp. 2623-2631).
        """

        from sklearn.model_selection import cross_val_score

        # Preprocess data
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Settings shared by every trial and by the final refit. See the docstring for
        # why the first two are not optional.
        fixed = {
            'random_state': self._seed,
            'bootstrap_type': 'Bernoulli',
            'allow_writing_files': False,
            'verbose': False,
        }

        # Define objective function for Optuna
        def objective(trial):
            """Optuna objective function for hyperparameter optimization."""
            # Suggest hyperparameters
            params = {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-2, 10.0, log=True),
            }

            # Create model with suggested parameters
            model = CatBoostRegressor(**params, **fixed)

            # Evaluate with cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )

            return scores.mean()

        # Create Optuna study with pruning
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self._seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Optimize hyperparameters
        study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

        # Get best parameters
        best_params = study.best_params
        best_params.update(fixed)

        # Train final model with best parameters
        best_model = CatBoostRegressor(**best_params)
        best_model.fit(X_train, y_train)

        # Make predictions
        preds = best_model.predict(X_test)

        # Evaluate on test set
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        result = {
            'fit_model': best_model,
            'preds': preds,
            'y_test': y_test,
            'params': best_params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'study': study  # Include study for analysis
        }

        return result


    def _sage_random_forest(self, X_train, X_test, y_train, y_test, n_iter=50, cv=5):
        """
        Train a Random Forest regressor as a sub-sage predictor with hyperparameter tuning.
        
        This function performs a randomized search over the hyperparameter space to find
        the best Random Forest configuration, then makes predictions on the test set.
        The search uses cross-validation to evaluate different parameter combinations.
        
        The function is called internally by :meth:`train_sub_sages` and is not meant
        to be called directly by users. It is designed to work with preprocessed data
        that has been split into training and test sets.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (data complexity metrics).
        X_test : pd.DataFrame
            Test features (data complexity metrics).
        y_train : pd.Series
            Training labels (model performance values).
        y_test : pd.Series
            Test labels (model performance values).
        n_iter : int, optional
            Number of iterations for randomized hyperparameter search. Default is 50.
            Higher values explore more parameter combinations but take longer.
        cv : int, optional
            Number of cross-validation folds for hyperparameter evaluation. Default is 5.
            Each parameter combination is evaluated using k-fold cross-validation.
        
        Returns
        -------
        dict
            Dictionary containing:
            
            - 'fit_model' : RandomizedSearchCV
                Trained Random Forest model with best parameters
            - 'preds' : np.ndarray
                Predictions on test set
            - 'y_test' : pd.Series
                True test labels
            - 'params' : dict
                Best hyperparameters found by randomized search
            - 'mae' : float
                Mean Absolute Error on test set
            - 'mse' : float
                Mean Squared Error on test set
            - 'rmse' : float
                Root Mean Squared Error on test set
            - 'r2' : float
                R² score on test set
        
        Notes
        -----
        The hyperparameter search space includes:
        
        - n_estimators: [100, 200, ..., 900] trees
        - max_depth: [5, 6, ..., 19] maximum tree depth
        - min_samples_split: [2, 3, ..., 9] minimum samples to split
        - min_samples_leaf: [1, 2, 3, 4] minimum samples per leaf
        - bootstrap: [True, False] whether to use bootstrap sampling
        
        The function handles infinite values and NaN by replacing them with 0.
        
        See Also
        --------
        _sage_mlp : Alternative MLP sub-sage
        train_sub_sages : Main training function that calls this method
        """

        param_distributions = {
            'n_estimators': np.arange(100, 1000, 100),
            'max_depth': np.arange(5, 20),
            'min_samples_split': np.arange(2, 10),
            'min_samples_leaf': np.arange(1, 5),
            'bootstrap': [True, False]
        }

        # Initialize the Random Forest Regressor
        rf = RandomForestRegressor(random_state=self._seed)

        # Initialize RandomizedSearchCV with configurable cv parameter
        rf_random = RandomizedSearchCV(estimator=rf,
                                    param_distributions=param_distributions,
                                    n_iter=n_iter,
                                    cv=cv,
                                    random_state=self._seed,
                                    n_jobs=-1)
        
        # Train
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        rf_random.fit(X_train, y_train)
        preds = rf_random.predict(X_test)
        params = rf_random.best_params_

        # Evaluate on held out
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        result = {
            'fit_model' : rf_random,
            'preds' : preds,
            'y_test' : y_test,
            'params' : params,
            'mae' : mae,
            'mse' : mse,
            'rmse' : rmse,
            'r2' : r2
        }

        return result

    @staticmethod
    def _save_plot(fig, saveFile, suffix):
        """Write ``fig`` next to ``saveFile`` with ``suffix`` before the extension.

        Replaces ``re.sub('.pdf', '', saveFile) + suffix + '.pdf'``, where the
        unescaped ``.`` was a regex wildcard: it stripped any character followed by
        ``pdf``, so ``sage_pdf_run.pdf`` became ``sage_run``. It also forced a
        ``.pdf`` extension regardless of what the caller asked for, so requesting
        ``results.png`` silently produced PDFs named ``results.png_..._barplot.pdf``.
        """
        root, ext = os.path.splitext(saveFile)
        ext = ext or '.pdf'
        path = f"{root}{suffix}{ext}"
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        fig.savefig(path, bbox_inches='tight')
        logger.info("Plot saved to: %s", path)
        return path

    def plot_results(self, figsize = (6,4), saveFile='', show=None ):

        ''' This function plots the results of the sub-sages trained on the input data.
        It will create a bar plot for each metric showing the performance of each model, and a scatter plot of the predictions vs. true values.
        The bar plot will show the mean absolute error (mae), mean squared error (mse), root mean squared error (rmse), and R2 score (r2) for each model.
        The scatter plot will show the predictions vs. true values for each model.
        If saveFile is provided, the plots will be saved to that file. Otherwise, the plots will be shown.
        It is designed to be used after the train_sub_sages function has been called, and the sub-sages have been trained.

        Args:
            figsize (tuple): Size of each figure.
            saveFile (str): Base file name for the plots. One bar plot and one
                scatter plot are written per metric, with ``_<metric>_barplot`` and
                ``_<metric>_scatterplot`` inserted before the extension. If empty,
                nothing is written. Default is ''.
            show (bool | None): Whether to call ``plt.show()``. ``None`` (the
                default) means "show only when not saving", which is what this
                docstring has always described; ``plt.show()`` used to be called
                unconditionally, so a run that saved to disk *also* blocked on a
                window under a GUI backend. It is ignored under a non-interactive
                backend either way.
        Returns:
            list[matplotlib.figure.Figure]: the figures, in the order drawn. All
            are already closed but remain savable.
        
        '''

        results = []
        preds = pd.DataFrame()
        for metric in self._available_metrics:
            for model in self._available_models:
                scores = pd.Series( self._results_subsages[metric][model].values(), index=self._results_subsages[metric][model].keys())
                results.append( [model, metric]+list(scores[['mae','mse','rmse','r2']]) )
                p = self._results_subsages[metric][model]['preds']
                y = self._results_subsages[metric][model]['y_test']
                preds = pd.concat( [preds,
                                    pd.DataFrame( [[model]*len(p),[metric]*len(p),p,y], index = ['model', 'metric', 'pred', 'y_test'] ).transpose() ] )
        
        # Create results DataFrame after collecting all results
        if not results:
            # Not a print: this is a library method, and a caller checking the
            # return value gets the same information from an empty list.
            logger.warning("No results to plot. Train the QSages first.")
            return []

        if show is None:
            show = saveFile == ''
        show = bool(show) and _can_show()
        figures = []
        
        results_df = pd.DataFrame(results, columns=['model','metric','mae','mse','rmse','r2'])
        results_df = results_df.melt(id_vars=['model', 'metric'])
        
        for metric in self._available_metrics:
            fig, ax = plt.subplots(figsize=figsize)
            # Filter data for current metric
            metric_data = results_df[results_df['metric']==metric]
            sns.barplot(data = metric_data, x = 'variable', y = 'value', hue = 'model', hue_order=self._available_models, ax=ax)  # type: ignore[arg-type]
            ax.set_title( "Predictive performance for each model for " + metric)
            ax.set_xlabel( "Metric")
            ax.set_ylabel( "Value" )
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
            if saveFile != '':
                self._save_plot(fig, saveFile, f'_{metric}_barplot')
            if show:
                plt.show()
            plt.close(fig)
            figures.append(fig)

            # Filter predictions for current metric
            toPlot = preds[ preds['metric'] == metric ]
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title( "Predictive performance for each model for " + metric)
            sns.scatterplot( data = toPlot, x = 'y_test', y = 'pred', hue = 'model', ax=ax )  # type: ignore[arg-type]
            ax.set_xlabel( "Actual")
            ax.set_ylabel( "Predicted" )
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
            if saveFile != '':
                self._save_plot(fig, saveFile, f'_{metric}_scatterplot')
            if show:
                plt.show()
            plt.close(fig)
            figures.append(fig)

        return figures


    def set_seed(self, seed=42):
        self._seed = seed


def calculate_SLGH(df, train_pct = 0.7):
    """Append the derived SLGH (Scaled Latent Geometric Hardness) feature.

    ``SLGH = -log(intrinsic_dim) - log(1 + FDR * n_train)``.

    Both ``Intrinsic_Dimension`` and ``Fisher Discriminant Ratio`` are computed
    natively by :mod:`qbiocode.evaluation.dataset_evaluation` in *both* schemas, so
    the formula is unchanged by the pyMFE integration. Only the sample count moved
    (``# Samples`` -> ``mfe.nr_inst``), which is why the column is looked up rather
    than named.

    Note that ``Fisher Discriminant Ratio`` is deliberately not replaced by pyMFE's
    ``mfe.f1.mean``: pyMFE's F1 is univariate and inverted (larger means *harder*),
    so substituting it would flip the sign of the second term. See the
    ``dataset_evaluation`` module docstring.

    Args:
        df (pd.DataFrame): Dataset-complexity features, one row per dataset.
        train_pct (float): Fraction of samples used for training, for ``n_train``.

    Returns:
        pd.DataFrame: A copy of ``df`` with the ``SLGH`` column appended.

    Raises:
        ValueError: If the sample-count column is absent in either schema.
    """
    id_col = 'Intrinsic_Dimension'
    fdr_col = 'Fisher Discriminant Ratio'
    num_samples = next(
        (name for name in SAMPLE_COUNT_COLUMN.values() if name in df.columns), None
    )
    if num_samples is None:
        raise ValueError(
            "Cannot derive SLGH: no sample-count column found. Expected one of "
            f"{sorted(SAMPLE_COUNT_COLUMN.values())}. Pass the complexity feature "
            "columns of a QProfiler results table."
        )
    n_train = np.ceil(df[num_samples] * train_pct)
    eps = 1e-8

    df = pd.DataFrame(df)
    df['SLGH'] = (-np.log(df[id_col] + eps) - np.log(1.0 + df[fdr_col] * n_train))
    return(df)


def main():
    """
    Command-line interface for QSage (Quantum Sage).
    
    This CLI allows users to train QSage models from the command line using CSV data files.
    QSage learns relationships between dataset complexity measures and model performance,
    enabling prediction of model performance on new datasets.
    
    Usage:
        qsage --input data.csv --output results/ [options]
    
    The input CSV should contain:
        - Dataset complexity features (Intrinsic_Dimension, Fisher Discriminant Ratio,
          and the mfe.* pyMFE block; a legacy-schema table is also accepted)
        - Performance metrics (accuracy, f1_score, auc)
        - Metadata (Dataset, embeddings, model, etc.)
    
    QProfiler Integration:
        QSage is designed to work directly with QProfiler output. Simply use the
        compiled_results.csv file generated by QProfiler as input:
        
        # Step 1: Run QProfiler
        qprofiler --config-name=config.yaml
        
        # Step 2: Train QSage with QProfiler output
        qsage --input compiled_results.csv --output sage_results/
    
    Examples:
        # Basic usage with QProfiler output
        qsage --input compiled_results.csv --output sage_results/
        
        # With custom cross-validation and hyperparameter search
        qsage --input compiled_results.csv --output results/ --cv 10 --n-iter 100
        
        # Train Random Forest sub-sages
        qsage --input data.csv --output results/ --model-type rf --seed 42
        
        # Train MLP sub-sages
        qsage --input data.csv --output results/ --model-type mlp --n-iter 2000
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='QSage: Quantum-inspired model selection oracle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Train QSage on profiler results
        qsage --input compiled_results.csv --output sage_results/
        
        # Train with a specific model type and iteration budget
        qsage --input data.csv --output results/ --model-type mlp --n-iter 200
        
        # Train with custom seed
        qsage --input data.csv --output results/ --seed 123
        
        For more information, see: https://qiskit-community.github.io/QBioCode/apps/sage.html
                """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input CSV file containing dataset features and model performance metrics'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for results and plots'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--model-type',
        default='random_forest',
        choices=['rf', 'mlp', 'random_forest', 'xgboost', 'xgboost_optuna',
                 'catboost', 'catboost_optuna'],
        help='Type of sub-sage model to train: rf/random_forest (Random Forest), mlp (MLP), '
             'xgboost/xgboost_optuna (XGBoost with Optuna - state-of-the-art), '
             'or catboost/catboost_optuna (CatBoost with Optuna). '
             'Default: random_forest. Only one type can be trained per run.'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--n-iter',
        type=int,
        default=None,
        help='For Random Forest: number of hyperparameter search iterations (default: 50). '
             'For MLP: maximum training epochs (default: 1000). '
             'For XGBoost-Optuna: number of Optuna trials (default: 100)'
    )
    
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds for hyperparameter evaluation (default: 5)'
    )
    
    args = parser.parse_args()

    # Validate every argument before doing any work: creating the output
    # directory, reading the CSV and training all take time or leave artifacts
    # behind, and a --cv=1 typo previously surfaced several minutes in as an
    # sklearn error about n_splits.
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    if os.path.isdir(args.input):
        print(
            f"Error: --input must be a CSV file, but '{args.input}' is a directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not 0.0 < args.test_size < 1.0:
        print(
            f"Error: --test-size is a proportion and must be strictly between 0 "
            f"and 1; got {args.test_size}.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.cv < 2:
        print(
            f"Error: --cv is the number of cross-validation folds and must be at "
            f"least 2; got {args.cv}.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.n_iter is not None and args.n_iter < 1:
        print(
            f"Error: --n-iter must be a positive integer (search iterations, "
            f"epochs or Optuna trials depending on --model-type); got "
            f"{args.n_iter}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("="*80)
    print("QSage: Quantum Model Selection Oracle")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Test size: {args.test_size}")
    print(f"Random seed: {args.seed}")
    print(f"Model type: {args.model_type}")
    print(f"Cross-validation folds: {args.cv}")
    if args.n_iter is not None:
        if args.model_type in ['rf', 'random_forest']:
            print(f"Hyperparameter search iterations: {args.n_iter}")
        elif args.model_type == 'mlp':
            print(f"Maximum training epochs: {args.n_iter}")
        elif args.model_type in ['xgboost', 'xgboost_optuna',
                                 'catboost', 'catboost_optuna']:
            print(f"Optuna optimization trials: {args.n_iter}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    try:
        data = pd.read_csv(args.input)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as e:
        print(f"Error reading {args.input}: {type(e).__name__}: {e}", file=sys.stderr)
        logger.debug("Failed to read %s", args.input, exc_info=True)
        sys.exit(1)
    if 'embeddings' not in data.columns:
        # A missing column is a malformed input, not a read failure; conflating the
        # two sent users looking for a filesystem problem that was not there.
        print(
            f"Error: {args.input} has no 'embeddings' column. "
            f"Found: {list(data.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)
    data['embeddings'] = ['None' if str(x) == 'nan' else x for x in data['embeddings']]
    print(f"Loaded {len(data)} rows with {len(data.columns)} columns")
    
    # Initialize QSage
    print("\nInitializing QSage...")
    try:
        sage = QuantumSage(data)
        sage.set_seed(args.seed)
        print(f"Available models: {sage._available_models}")
        print(f"Available embeddings: {sage._available_embeddings}")
        print(f"Available metrics: {sage._available_metrics}")
    except Exception as e:
        # Broad by design: this is a CLI boundary, where an unexpected failure has
        # to become an exit code and a readable line rather than a traceback. The
        # traceback is kept at DEBUG so `-v` / a logging config can still get it.
        print(f"Error initializing QSage: {type(e).__name__}: {e}", file=sys.stderr)
        logger.debug("QSage initialization failed", exc_info=True)
        sys.exit(1)
    
    # Train sub-sages
    print(f"\nTraining sub-sages...")
    try:
        # Map CLI argument to sage_type
        sage_type_map = {
            'rf': 'random_forest',
            'random_forest': 'random_forest',
            'mlp': 'mlp',
            'xgboost': 'xgboost_optuna',
            'xgboost_optuna': 'xgboost_optuna',
            'catboost': 'catboost_optuna',
            'catboost_optuna': 'catboost_optuna'
        }
        sage_type = sage_type_map[args.model_type]
        
        print(f"  Training {sage_type} sub-sages...")
        sage.train_sub_sages(
            test_size=args.test_size,
            sage_type=sage_type,
            n_iter=args.n_iter,
            cv=args.cv
        )
        
        print("Training complete!")
    except Exception as e:
        print(f"Error training sub-sages: {type(e).__name__}: {e}", file=sys.stderr)
        logger.debug("Sub-sage training failed", exc_info=True)
        sys.exit(1)

    # Save Sage model. `with` so the handle is closed even on a pickling error --
    # pickle.dump(obj, open(...)) leaked the file object and, on failure, left a
    # truncated .pkl behind with no indication that it was incomplete.
    model_path = os.path.join(args.output, 'trained_sage.pkl')
    try:
        with open(model_path, 'wb') as fh:
            pickle.dump(sage, fh)
    except (OSError, pickle.PicklingError, TypeError, AttributeError) as e:
        print(f"Error saving the trained model to {model_path}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        logger.debug("Failed to pickle the trained sage", exc_info=True)
        sys.exit(1)

    # Generate and save plots
    print(f"\nGenerating plots...")
    try:
        output_file = os.path.join(args.output, 'sage_results.pdf')
        sage.plot_results(saveFile=output_file)
        print(f"Plots saved to: {output_file}")
    except Exception as e:
        # Non-fatal: plots are a convenience, and the results summary below is the
        # actual output. Warn and continue rather than discarding a completed run.
        print(f"Warning: could not generate plots: {type(e).__name__}: {e}", file=sys.stderr)
        logger.debug("Plot generation failed", exc_info=True)
    
    # Save results summary
    print("\nSaving results summary...")
    try:
        results_summary = []
        for metric in sage._available_metrics:
            for model in sage._available_models:
                if metric in sage._results_subsages and model in sage._results_subsages[metric]:
                    result = sage._results_subsages[metric][model]
                    results_summary.append({
                        'model': model,
                        'metric': metric,
                        'mae': result['mae'],
                        'mse': result['mse'],
                        'rmse': result['rmse'],
                        'r2': result['r2']
                    })
        
        if results_summary:
            results_df = pd.DataFrame(results_summary)
            results_file = os.path.join(args.output, 'sage_summary.csv')
            results_df.to_csv(results_file, index=False)
            print(f"Results summary saved to: {results_file}")
            print("\nResults Summary:")
            print(results_df.to_string(index=False))
        else:
            print("Warning: No results to save")
    except Exception as e:
        print(f"Warning: could not save the results summary: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        logger.debug("Failed to write the results summary", exc_info=True)
    
    print("\n" + "="*80)
    print("QSage training completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
