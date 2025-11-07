import logging
from collections import defaultdict

import pandas as pd
import numpy as np
import numpy.linalg as npla
import torch
from torch import nn
from random import shuffle
import math
from xgboost import XGBRegressor
import json
from pprint import pprint
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split

import autograd_hacks
from base.agent import Agent
from utils_nn import Model
from utils_xgboost import get_leafs, XGBModel

import sklearn.model_selection as skm
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


class BernoulliLinUCB(Agent):
    """LinUCB for Bernoulli rewards"""

    def __init__(self, env_constructor,
                 context_size, use_cuda=True,
                 alpha=1.0):

        self.t = 1
        self.context_size = context_size
        self.alpha = alpha
        self.iteration_data = dict()

        self.internal_env = env_constructor()

        self.use_cuda = use_cuda

        self.b_vectors = defaultdict()
        self.A_matrices = defaultdict()
        for arm in self.internal_env.arm_set:
                self.b_vectors[arm] = np.zeros(self.context_size).reshape((self.context_size, 1))
                self.A_matrices[arm] = np.identity(self.context_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        Agent.__init__(self)

    
    def get_upper_confidence_bound(self, observation):

        ucbs = defaultdict()

        temp_list = []
        logging.debug("observation", observation)
        for arm in observation:
            logging.debug("observation[arm]", observation[arm])
            x = observation[arm].context
            b = self.b_vectors[arm]
            A = self.A_matrices[arm]
            try:
                A_inv = np.linalg.inv(A)
            except:
                raise ValueError("A_inv matrix has NaN values!")
            mu = np.dot(np.matmul(A_inv, b).T, x)
            cb = self.alpha * np.sqrt(np.dot(x.T, np.matmul(A_inv, x)))
            ucbs[arm] = mu + cb

        return ucbs

    def inv_sherman_morrison(self, u, A_inv):
        """Inverse of a matrix with rank 1 update"""

        Au = np.dot(A_inv, u)
        A_inv -= np.outer(Au, Au) / (1 + np.dot(u.T, Au))
        return A_inv

    def update_observation(self, observation, action, reward):
        """Apply Ridge regression"""

        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array([reward]).reshape((1, 1))  
        self.b_vectors[action.label] += y * x
        self.A_matrices[action.label] += np.outer(x, x.T)

        return

    def pick_action(self, observation):

        ucbs = self.get_upper_confidence_bound(observation)
        self.internal_env.overwrite_arm_weight(ucbs)
        arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1
        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data


class BernoulliLinTS(Agent):
    """LinTS for Bernoulli rewards"""

    def __init__(self, env_constructor,
                 context_size, use_cuda=True,
                 nu=1.0):

        self.t = 1
        self.context_size = context_size
        self.nu = nu
        self.iteration_data = dict()

        self.internal_env = env_constructor()

        self.use_cuda = use_cuda

        self.f_vectors = defaultdict()
        self.B_matrices = defaultdict()
        for arm in self.internal_env.arm_set:
                self.f_vectors[arm] = np.zeros(self.context_size).reshape((self.context_size, 1))
                self.B_matrices[arm] = np.identity(self.context_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        Agent.__init__(self)

    def get_sampled_rewards(self, observation):

        sampled_mean_rewards = defaultdict()

        temp_list = []
        logging.debug("observation", observation)
        for arm in observation:
            logging.debug("observation[arm]",observation[arm])
            x = observation[arm].context
            f = self.f_vectors[arm]
            B = self.B_matrices[arm]
            try:
                B_inv = np.linalg.inv(B)
            except:
                raise ValueError("B_inv matrix has NaN values!")
            mu_hat = np.matmul(B_inv, f)
            mu_tilde = np.random.multivariate_normal(mu_hat.flatten(), self.nu**2*B_inv)
            mean_reward = np.dot(x.T, mu_tilde)
            
            sampled_mean_rewards[arm] = mean_reward

        return sampled_mean_rewards

    def inv_sherman_morrison(self, u, A_inv):
        """Inverse of a matrix with rank 1 update"""
        Au = np.dot(A_inv, u)
        A_inv -= np.outer(Au, Au) / (1 + np.dot(u.T, Au))
        return A_inv

    def update_observation(self, observation, action, reward):
        """Apply Ridge regression"""

        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array([reward]).reshape((1, 1))  
        self.f_vectors[action.label] += y * x
        self.B_matrices[action.label] += np.outer(x, x.T)

        return

    def pick_action(self, observation):

        samples = self.get_sampled_rewards(observation)
        self.internal_env.overwrite_arm_weight(samples)
        arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1
        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data


class BernoulliNeuralUCB(Agent):
    """NeuralUCB for Bernoulli rewards"""

    def __init__(self, env_constructor,
                 context_size, n_layers, hidden_size,
                 reg_factor, delta, confidence_scaling_factor,
                 dropout_probability, learning_rate, epochs, batch_size_factor,
                 replay_buffer_size, early_stopping, UCB_extra_t_factor,
                 use_cuda=True, initial_random_selections=10):

        self.t = 1
        self.context_size = context_size

        self.internal_env = env_constructor()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size_factor = batch_size_factor
        self.replay_buffer_size = replay_buffer_size
        self.early_stopping = early_stopping
        self.dropout_probability = dropout_probability
        self.reg_factor = reg_factor
        self.delta = delta
        self.confidence_scaling_factor = confidence_scaling_factor
        self.UCB_extra_t_factor = UCB_extra_t_factor
        self.use_cuda = use_cuda
        self.confidence_scaling_factor = confidence_scaling_factor
        self.initial_random_selections = initial_random_selections

        self.x_train = None
        self.reward_theta = None
        print_params=False
        if print_params:
            print("n_layers =", str(self.n_layers))
            print("hidden_size", str(self.hidden_size))
            print("learningrate",str(self.learning_rate))
            print("epochs",str(self.epochs))
            print("buffer_size",str(self.replay_buffer_size))
            print("early_stopping",str(self.early_stopping))
            print("drop_prob",str(self.dropout_probability))
            print("reg_factor",str(self.reg_factor))
            print("delta",str(self.delta))
            print("confidence_scaling_factor",str(confidence_scaling_factor))
            print("use_cuda",str(use_cuda))


        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        logging.info('torch.cuda.is_available():' + str(torch.cuda.is_available()) + ', self.use_cuda:' + str(self.use_cuda) + ', self.device:' + str(self.device))

        self.model = Model(input_size=self.context_size,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           dropout_prob=self.dropout_probability
                           ).to(self.device)

        parameterList = [param for param in self.model.parameters()]
        self.theta_0 = torch.cat([param.flatten() for param in parameterList], axis=0).cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.bound_features = 1
        self.theta_d = sum(w.numel() for w in self.model.parameters() if w.requires_grad)
        self.A = self.reg_factor * torch.eye(self.theta_d).to(self.device)
        self.A_inv_diag = 1 / torch.diag(self.A)
        self.iteration_data = dict()

        self.grad_approxs = dict()
        for arm in self.internal_env.arm_set:
                self.grad_approxs[arm] = torch.zeros((self.theta_d,)).to(self.device)

        autograd_hacks.add_hooks(self.model)

        Agent.__init__(self)

    def train_model(self, observation, action, reward):
        """Train neural network"""

        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array(reward).reshape((1, 1))
        if self.x_train is not None:
            self.x_train = np.append(self.x_train, x, axis=1)
        else:
            self.x_train = x
        if self.reward_theta is not None:
            self.reward_theta = np.append(self.reward_theta, y, axis=0)
        else:
            self.reward_theta = y

        #Mini-Batch Gradient Descent
        self.model.train()
        n_dps = min(len(self.reward_theta), self.replay_buffer_size)
        batch_size = max(1, int(n_dps / self.batch_size_factor))
        for _ in range(self.epochs):
            #sample training batches
            indices = np.random.choice(n_dps, n_dps, replace=False)
            train_loss = 0
            #loop through mini-batches
            for mb in range(0, int(n_dps / batch_size)):
                batch_indices = indices[mb*batch_size:(mb+1)*batch_size]
                x_train_tensor = torch.FloatTensor(self.x_train[:,batch_indices]).to(self.device)
                reward_theta_tensor = torch.FloatTensor(self.reward_theta[batch_indices]).to(self.device)
                y_pred = self.model(x_train_tensor.T)
                parameterList = [param for param in self.model.parameters()]
                self.theta_current = torch.cat([param.flatten() for param in parameterList], axis=0).cpu()
                loss = nn.MSELoss()(reward_theta_tensor, y_pred) + self.hidden_size * self.reg_factor * torch.sum(torch.square(self.theta_current - self.theta_0)) / 2
                train_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return train_loss.item()

    def predict(self, x):
        """Predict rewards """

        # eval mode
        self.model.eval()
        mu_hat = self.model(torch.FloatTensor(x.T).to(self.device)).detach().squeeze()
        return mu_hat.cpu().numpy()

    def update_output_gradient(self, observation):
        """Get gradient of network prediction w.r.t network weights for all arms at once.
        Actual gradients calculated in get_upper_confidence_bound.
        """

        self.x_grad_np = None
        for arm in self.grad_approxs:
            x = observation[arm].context
            if self.x_grad_np is not None:
                self.x_grad_np = np.append(self.x_grad_np, x, axis=1)
            else:
                self.x_grad_np = x

        return

    def confidence_multiplier(self, t):
        """NeuralUCB confidence interval multiplier"""

        factor = self.confidence_scaling_factor * np.sqrt(
            self.theta_d * np.log(
                1 + t * self.bound_features ** 2 / (self.reg_factor * self.theta_d)
            ) + 2 * np.log(1 / self.delta)
        )
        factor *= t**self.UCB_extra_t_factor
        return factor
    
    def get_upper_confidence_bound(self, observation, t):
        ucbs = dict()
        self.update_output_gradient(observation)

        temp_list = []

        self.model.zero_grad()
        autograd_hacks.clear_backprops(self.model)
        x_grad = torch.FloatTensor(self.x_grad_np).to(self.device)
        predictions = self.model(x_grad.T)
        pred_loss = torch.sum(predictions)
        pred_loss.backward()
        autograd_hacks.compute_grad1(self.model)

        dp_idx = 0
        for arm in self.grad_approxs:
            grad_approx = torch.cat([param.grad1[dp_idx].flatten() for param in self.model.parameters() if param.requires_grad])#.cpu().numpy()
            self.grad_approxs[arm] = grad_approx
            exploration_bonus = torch.dot(grad_approx, torch.mul(self.A_inv_diag, grad_approx)).item() / self.hidden_size

            if torch.isnan(grad_approx).any():
                raise ValueError("Gradient approximations have NaN values!")

            a = predictions[dp_idx].item()

            #UCB as in NeuralTS repo
            b = np.sqrt(self.reg_factor * self.confidence_scaling_factor)
            c = np.sqrt(exploration_bonus)
            #ucb = max(1e-12, a + b * c)
            ucb = a + b * c
            ucbs[arm] = ucb

            temp_list.append((a,b,c,ucb))
            dp_idx += 1

        logging.debug('a: ' + str(a) + ', b: ' + str(b) + ', c: ' + str(c) + ', ucb: ' + str(ucb) + ', ||Grad_approx||: ' + str(torch.norm(grad_approx)))
        
        return ucbs

    def inv_diagonal(self, u, A):
        """Approx. inverse of matrix A by keeping only diagonal elements (like NeuralUCB and NeuralTS papers)"""
        G_tensor = torch.outer(u, u)
        A += G_tensor / self.hidden_size
        A_inv_diag = 1 / torch.diag(A)
        return A, A_inv_diag

    def update_observation(self, observation, action, reward):
        train_loss = self.train_model(observation, action, reward)
        self.iteration_data['train_loss'] = train_loss
        g = self.grad_approxs[action.label]
        self.A, self.A_inv_diag = self.inv_diagonal(g, self.A)
        if torch.isnan(self.A_inv_diag).any():
            raise ValueError("A_inv matrix has NaN values!")
        return

    def pick_action(self, observation):

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
        else:
            ucbs = self.get_upper_confidence_bound(observation, self.t)
            self.internal_env.overwrite_arm_weight(ucbs)
            arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1
        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data


class BernoulliNeuralTS(Agent):
    """NeuralTS for Bernoulli rewards"""

    def __init__(self, env_constructor,
                 context_size, n_layers, hidden_size,
                 reg_factor, exploration_var, dropout_probability, 
                 learning_rate, epochs, batch_size_factor,
                 replay_buffer_size, early_stopping, use_cuda=True,
                 initial_random_selections=10):

        self.t = 1
        self.context_size = context_size

        self.internal_env = env_constructor()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size_factor = batch_size_factor
        self.replay_buffer_size = replay_buffer_size
        self.dropout_probability = dropout_probability
        self.reg_factor = reg_factor
        self.exploration_var = exploration_var
        self.early_stopping = early_stopping
        self.use_cuda = use_cuda
        self.initial_random_selections = initial_random_selections

        self.x_train = None
        self.reward_theta = None

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        logging.info('torch.cuda.is_available():' + str(torch.cuda.is_available()) + ', self.use_cuda:' + str(self.use_cuda) + ', self.device:' + str(self.device))

        self.model = Model(input_size=self.context_size,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           dropout_prob=self.dropout_probability
                           ).to(self.device)

        parameterList = [param for param in self.model.parameters()]
        self.theta_0 = torch.cat([param.flatten() for param in parameterList], axis=0).cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.bound_features = 1
        self.theta_d = sum(w.numel() for w in self.model.parameters() if w.requires_grad)
        self.A = self.reg_factor * torch.eye(self.theta_d).to(self.device)
        self.A_inv_diag = 1 / torch.diag(self.A)
        self.iteration_data = dict()

        self.grad_approxs = dict()
        for arm in self.internal_env.arm_set:
                self.grad_approxs[arm] = torch.zeros((self.theta_d,)).to(self.device)

        autograd_hacks.add_hooks(self.model)

        Agent.__init__(self)

    def train_model(self, observation, action, reward):
        """Train neural network"""

        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array(reward).reshape((1, 1))
        if self.x_train is not None:
            self.x_train = np.append(self.x_train, x, axis=1)
        else:
            self.x_train = x
        if self.reward_theta is not None:
            self.reward_theta = np.append(self.reward_theta, y, axis=0)
        else:
            self.reward_theta = y
        
        #Mini-Batch Gradient Descent
        self.model.train()
        n_dps = min(len(self.reward_theta), self.replay_buffer_size)
        batch_size = max(1, int(n_dps / self.batch_size_factor))
        for _ in range(self.epochs):
            #sample training batches
            indices = np.random.choice(n_dps, n_dps, replace=False)
            train_loss = 0
            #loop through mini-batches
            for mb in range(0, int(n_dps / batch_size)):
                batch_indices = indices[mb*batch_size:(mb+1)*batch_size]
                x_train_tensor = torch.FloatTensor(self.x_train[:,batch_indices]).to(self.device)
                reward_theta_tensor = torch.FloatTensor(self.reward_theta[batch_indices]).to(self.device)
                y_pred = self.model(x_train_tensor.T)
                parameterList = [param for param in self.model.parameters()]
                self.theta_current = torch.cat([param.flatten() for param in parameterList], axis=0).cpu()
                loss = nn.MSELoss()(reward_theta_tensor, y_pred) + self.hidden_size * self.reg_factor * torch.sum(torch.square(self.theta_current - self.theta_0)) / 2
                train_loss += loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

        return train_loss.item()

    def predict(self, x):
        """Predict reward"""

        # eval mode
        self.model.eval()
        mu_hat = self.model(torch.FloatTensor(x.T).to(self.device)).detach().squeeze()
        return mu_hat.cpu().numpy()

    def update_output_gradient(self, observation):
        """Get gradient of network prediction w.r.t network weights for all arms at once.
        Actual gradients calculated in get_samples.
        """

        self.x_grad_np = None
        for arm in self.grad_approxs:
            x = observation[arm].context
            if self.x_grad_np is not None:
                self.x_grad_np = np.append(self.x_grad_np, x, axis=1)
            else:
                self.x_grad_np = x

        return

    def get_samples(self, observation, t):

        samples = dict()
        self.update_output_gradient(observation)

        self.model.zero_grad()
        autograd_hacks.clear_backprops(self.model)
        x_grad = torch.FloatTensor(self.x_grad_np).to(self.device)
        predictions = self.model(x_grad.T)
        pred_loss = torch.sum(predictions)
        pred_loss.backward()
        autograd_hacks.compute_grad1(self.model)

        dp_idx = 0
        for arm in self.grad_approxs:
            grad_approx = torch.cat([param.grad1[dp_idx].flatten() for param in self.model.parameters() if param.requires_grad])#.cpu().numpy()
            self.grad_approxs[arm] = grad_approx

            if torch.isnan(grad_approx).any():
                raise ValueError("Gradient approximations have NaN values!")

            variance = self.reg_factor * torch.dot(grad_approx.T, torch.mul(self.A_inv_diag, grad_approx)).item() / self.hidden_size
            mean = predictions[dp_idx].item()
            sample = np.random.normal(mean, self.exploration_var * np.sqrt(variance))
            samples[arm] = sample

            dp_idx += 1

        logging.debug('mean: ' + str(mean) + ', std dev.: ' + str(np.sqrt(variance)) + ', expl. var.: ' + str(self.exploration_var) + ', sample: ' + str(sample) + ', ||Grad_approx||: ' + str(torch.norm(grad_approx)))
        
        return samples

    def inv_diagonal(self, u, A):
        """Approx. inverse of matrix A by keeping only diagonal elements (like NeuralUCB and NeuralTS papers)"""
        G_tensor = torch.outer(u, u)
        A += G_tensor / self.hidden_size
        A_inv_diag = 1 / torch.diag(A)
        return A, A_inv_diag

    def update_observation(self, observation, action, reward):
        train_loss = self.train_model(observation, action, reward)
        self.iteration_data['train_loss'] = train_loss

        # NOTE: I did not re-calculate grad_approxs again
        g = self.grad_approxs[action.label]
        self.A, self.A_inv_diag = self.inv_diagonal(g, self.A)
        if torch.isnan(self.A_inv_diag).any():
            raise ValueError("A_inv matrix has NaN values!")
        return

    def pick_action(self, observation):

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
        else:
            samples = self.get_samples(observation, self.t)
            self.internal_env.overwrite_arm_weight(samples)
            arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1
        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data


class BernoulliXGBoostTEUCB(Agent):
    """TEUCB with XGBoost for Bernoulli rewards"""

    def __init__(self, env_constructor, context_size, 
                use_cuda=False, exploration_factor=1.0,
                max_depth=6, n_estimators=100,
                eta=0.3, gamma=0.0, xgb_lambda=1.0,
                enable_categorical=True, initial_random_selections=10,
                feature_types=None, base_score=1):

        self.t = 1
        self.context_size = context_size

        self.internal_env = env_constructor()

        self.exploration_factor = exploration_factor
        self.use_cuda = use_cuda
        self.x_train = None
        self.reward_theta = None

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.lr = eta

        self.initial_random_selections = initial_random_selections
        self.retrain_check = 0

        #list of types (numeric or categorical) for each feature in x_train
        self.feature_types = feature_types
        self.base_score = base_score
        self.enable_categorical = enable_categorical

        print('XGBoostUCB parameters - exploration_factor:', exploration_factor, 'max_depth:', max_depth,
               'n_estimators:', n_estimators, 'eta:', eta, 'gamma:', gamma, 'lambda:', xgb_lambda, 'base_score:', base_score)
        
        self.model_parameters = {'booster': 'gbtree', 'tree_method': 'hist', 'objective': 'reg:squarederror', 'base_score': base_score, 
        'max_depth': max_depth, 'gamma': gamma, 'learning_rate': eta, 'reg_lambda': xgb_lambda, 'min_child_weight': 2}
        self.iteration_data = dict()
        Agent.__init__(self)

    def train_model(self, observation, action, reward):
        """Train XGBoost approximator"""

        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array(reward).reshape((1, 1))
        if self.x_train is not None:
            self.x_train = np.append(self.x_train, x, axis=1)
        else:
            self.x_train = x
        if self.reward_theta is not None:
            self.reward_theta = np.append(self.reward_theta, y, axis=0)
        else:
            self.reward_theta = y

        #re-train with increasing time interval, otherwise only update leaf values
        if self.t >= self.initial_random_selections and np.ceil(8*np.log(self.t)) > self.retrain_check:

            self.retrain_check = np.ceil(8*np.log(self.t))  #set to e.g. 0 to re-train every step

            X = self.x_train.T
            y = self.reward_theta
            #logging.info(str(self.x_train.T.shape) + '\n' + str(len(self.feature_types)))
            assert self.x_train.T.shape[1] == len(self.feature_types)
            Xy = xgb.DMatrix(X, y, feature_types=self.feature_types, enable_categorical=self.enable_categorical)

            self.model = xgb.train(params = self.model_parameters, dtrain=Xy, num_boost_round=self.n_estimators)

            logging.info('t = ' + str(self.t) + ' - re-trained model')

            leaf_scores = self.model.get_dump(with_stats=True, dump_format='json')

            X = xgb.DMatrix(self.x_train.T, feature_types=self.feature_types, enable_categorical=self.enable_categorical)
            residuals_array = np.array(self.reward_theta - self.base_score, dtype="float64")

            #traverse tree to get leafs, then read leaf data (score and cover)
            json_trees = [json.loads(leaf_scores[i]) for i in range(self.n_estimators)]
            self.leaves_per_tree = []
            for idx, tree in enumerate(json_trees):
                if not self.leaves_per_tree:  #always append leafs of first tree (even if only one leaf)
                    self.leaves_per_tree.append(get_leafs(tree))
                else:
                    self.leaves_per_tree.append(get_leafs(tree))

                pred = self.model.predict(X, iteration_range=(0,idx+1)).reshape((len(self.reward_theta),1))
                residuals_array = np.append(residuals_array, self.reward_theta - pred, axis=1)

            individual_preds = self.model.predict(X, pred_leaf=True)
            leaf_of_data = np.array(individual_preds)#.T

            for i in range(len(self.leaves_per_tree)):
                df = pd.DataFrame({'leaf':leaf_of_data[:,i], 'residual':residuals_array[:,i]})
                group = df.groupby(by='leaf').agg(['count', 'mean', 'var'])

                for row_idx, row in group.iterrows():
                    self.leaves_per_tree[i][row_idx]["row_idx"] = row_idx
                    self.leaves_per_tree[i][row_idx]["leaf_count"] = row['residual']['count']
                    self.leaves_per_tree[i][row_idx]["leaf_mean"] = row['residual']['mean'] * self.lr
                    self.leaves_per_tree[i][row_idx]["leaf_variance"] = row['residual']['var'] * self.lr**2

        elif self.t >= self.initial_random_selections:
            X = xgb.DMatrix(x.T, feature_types=self.feature_types, enable_categorical=self.enable_categorical)

            #get leaf indices
            leaf_preds = np.array(self.model.predict(X, pred_leaf=True))

            #get residual and update leaf values
            residual = y - self.base_score  #first residual
            for i in range(len(self.leaves_per_tree)):
                pred = self.model.predict(X, iteration_range=(0,i+1))

                self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_mean"] = ( (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_mean"] * 
                                                                        self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] + 
                                                                        residual * self.lr) / (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] + 1) )[0][0]
                self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_variance"] = ( (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_variance"] *
                                                                            (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] - 1) + 
                                                                            (pred - residual)**2 * self.lr**2) / (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"]) )[0][0]
                self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] += 1

                residual = y - pred

        return

    def predict(self, x):
        """Predict reward"""

        X = xgb.DMatrix(x.T)

        individual_preds = self.model.predict(X, pred_leaf=True)
        leaf_assignments = np.array(individual_preds)#.T

        n_samples_total = len(self.reward_theta)
        
        mean_per_assigned_leaf = np.array([self.leaves_per_tree[i][leaf_assignments[0][i]]["leaf_mean"] for i in range(len(self.leaves_per_tree))])

        mu_hat = self.base_score + np.sum(mean_per_assigned_leaf)

        variance_of_mean_per_assigned_leaf = np.array([self.leaves_per_tree[i][leaf_assignments[0][i]]["leaf_variance"] / self.leaves_per_tree[i][leaf_assignments[0][i]]["leaf_count"] for i in range(len(self.leaves_per_tree))])         
        total_variance_of_mean = np.sum(variance_of_mean_per_assigned_leaf)

        c = self.exploration_factor * np.sqrt(total_variance_of_mean * np.log(self.t - 1) / n_samples_total)

        return mu_hat + c

    def get_samples(self, observation, t):
        samples = dict()
        for arm in self.internal_env.arm_set:
            x = observation[arm].context
            pred = self.predict(x)
            samples[arm] = pred

        return samples


    def update_observation(self, observation, action, reward):
        self.train_model(observation, action, reward)

        return

    def pick_action(self, observation):

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
        else:
            samples = self.get_samples(observation, self.t)
            self.internal_env.overwrite_arm_weight(samples)
            arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1

        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data


class BernoulliXGBoostTETS(Agent):
    """TETS with XGBoost for Bernouilli rewrads"""

    def __init__(self, env_constructor, context_size, 
                use_cuda=False, exploration_variance=1.0,
                max_depth=6, n_estimators=100,
                eta=0.3, gamma=0.0, xgb_lambda=1.0, 
                enable_categorical=True, initial_random_selections=10, 
                feature_types=None, base_score=1):

        self.t = 1
        self.context_size = context_size

        self.internal_env = env_constructor()

        self.exploration_variance = exploration_variance
        self.use_cuda = use_cuda
        self.x_train = None
        self.reward_theta = None

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.lr = eta

        self.initial_random_selections = initial_random_selections
        self.retrain_check = 0
        self.base_score = base_score

        #list of types (numeric or categorical) for each feature in x_train
        self.feature_types = feature_types
        self.enable_categorical = enable_categorical

        print('XGBoostTS parameters - exploration_variance:', self.exploration_variance, 'max_depth:', max_depth,
               'n_estimators:', n_estimators, 'eta:', eta, 'gamma:', gamma, 'lambda:', xgb_lambda)
        
        self.model_parameters = {'booster': 'gbtree', 'tree_method': 'hist', 'objective': 'reg:squarederror', 'base_score': base_score, 'max_depth': max_depth, 'gamma': gamma, 'learning_rate': eta, 'reg_lambda': xgb_lambda, 'min_child_weight': 2}
        self.iteration_data = dict()
        Agent.__init__(self)

    def train_model(self, observation, action, reward):
        """Train XGBoost approximator"""

        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array(reward).reshape((1, 1))
        if self.x_train is not None:
            self.x_train = np.append(self.x_train, x, axis=1)
        else:
            self.x_train = x
        if self.reward_theta is not None:
            self.reward_theta = np.append(self.reward_theta, y, axis=0)
        else:
            self.reward_theta = y

        #re-train with increasing time interval, otherwise only update leaf values
        if self.t >= self.initial_random_selections and np.ceil(8*np.log(self.t)) > self.retrain_check:

            self.retrain_check = np.ceil(8*np.log(self.t))  #set to e.g. 0 to re-train every step

            X = self.x_train.T
            y = self.reward_theta
            assert self.x_train.T.shape[1] == len(self.feature_types)
            Xy = xgb.DMatrix(X, y, feature_types=self.feature_types, enable_categorical=self.enable_categorical)

            self.model = xgb.train(params = self.model_parameters, dtrain=Xy, num_boost_round=self.n_estimators)

            logging.info('t = ' + str(self.t) + ' - re-trained model')

            leaf_scores = self.model.get_dump(with_stats=True, dump_format='json')

            X = xgb.DMatrix(self.x_train.T, feature_types=self.feature_types, enable_categorical=self.enable_categorical)
            residuals_array = np.array(self.reward_theta - self.base_score, dtype="float64")

            #traverse tree to get leafs, then read leaf data (score and cover)
            json_trees = [json.loads(leaf_scores[i]) for i in range(self.n_estimators)]
            self.leaves_per_tree = []
            for idx, tree in enumerate(json_trees):
                if not self.leaves_per_tree:  #always append leafs of first tree (even if only one leaf)
                    self.leaves_per_tree.append(get_leafs(tree))

                else:
                    self.leaves_per_tree.append(get_leafs(tree))

                pred = self.model.predict(X, iteration_range=(0,idx+1)).reshape((len(self.reward_theta),1))

                residuals_array = np.append(residuals_array, self.reward_theta - pred, axis=1)

            individual_preds = self.model.predict(X, pred_leaf=True)
            leaf_of_data = np.array(individual_preds)#.T

            for i in range(len(self.leaves_per_tree)):
                df = pd.DataFrame({'leaf':leaf_of_data[:,i], 'residual':residuals_array[:,i]})
                group = df.groupby(by='leaf').agg(['count', 'mean', 'var'])

                for row_idx, row in group.iterrows():
                    self.leaves_per_tree[i][row_idx]["row_idx"] = row_idx
                    self.leaves_per_tree[i][row_idx]["leaf_count"] = row['residual']['count']
                    self.leaves_per_tree[i][row_idx]["leaf_mean"] = row['residual']['mean'] * self.lr
                    self.leaves_per_tree[i][row_idx]["leaf_variance"] = row['residual']['var'] * self.lr**2

        #if no re-training, update affected leafs with latest datapoint (x, y)
        elif self.t >= self.initial_random_selections:
            X = xgb.DMatrix(x.T, feature_types=self.feature_types, enable_categorical=self.enable_categorical)

            #get leaf indices
            leaf_preds = np.array(self.model.predict(X, pred_leaf=True))

            #get residual and update leaf values
            residual = y - self.base_score  #first residual
            for i in range(len(self.leaves_per_tree)):
                pred = self.model.predict(X, iteration_range=(0,i+1))

                self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_mean"] = ( (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_mean"] * 
                                                                        self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] + 
                                                                        residual * self.lr) / (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] + 1) )[0][0]
                self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_variance"] = ( (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_variance"] *
                                                                            (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] - 1) + 
                                                                            (pred - residual)**2 * self.lr**2) / (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"]) )[0][0]
                self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] += 1

                residual = y - pred

        return


    def predict(self, x):
        """Predict reward"""

        X = xgb.DMatrix(x.T)

        individual_preds = self.model.predict(X, pred_leaf=True)
        leaf_assignments = np.array(individual_preds)#.T
    
        mean_per_assigned_leaf = np.array([self.leaves_per_tree[i][leaf_assignments[0][i]]["leaf_mean"] for i in range(len(self.leaves_per_tree))])
        mu_hat = self.base_score + np.sum(mean_per_assigned_leaf)
        variance_of_mean_per_assigned_leaf = np.array([self.leaves_per_tree[i][leaf_assignments[0][i]]["leaf_variance"] / self.leaves_per_tree[i][leaf_assignments[0][i]]["leaf_count"] for i in range(len(self.leaves_per_tree))])         
        total_variance_of_mean = np.sum(variance_of_mean_per_assigned_leaf)

        return mu_hat + np.sqrt(total_variance_of_mean) * np.random.randn()


    def get_samples(self, observation, t):
        samples = dict()
        for arm in self.internal_env.arm_set:
            x = observation[arm].context
            pred = self.predict(x)
            samples[arm] = pred
        return samples


    def update_observation(self, observation, action, reward):
        self.train_model(observation, action, reward)
        return

    def pick_action(self, observation):

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
        else:
            samples = self.get_samples(observation, self.t)
            self.internal_env.overwrite_arm_weight(samples)
            arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1

        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data


class BernoulliRandomForestTEUCB(Agent):
    """TEUCB with Random Forest for Bernoulli rewards"""

    def __init__(self, env_constructor, context_size, 
                use_cuda=False, exploration_factor=1.0, max_depth=6,
                n_estimators=100, initial_random_selections=10):

        self.t = 1
        self.context_size = context_size

        self.internal_env = env_constructor()

        self.exploration_factor = exploration_factor
        self.use_cuda = use_cuda
        self.x_train = None
        self.reward_theta = None

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.max_depth = max_depth
        self.n_estimators = n_estimators

        self.initial_random_selections = initial_random_selections
        self.retrain_check = 0

        print('Random Forest parameters - exploration_factor:', self.exploration_factor, 'n_estimators:', self.n_estimators)
        
        self.model_parameters = {'min_child_weight': 2}
        self.iteration_data = dict()
        Agent.__init__(self)

    def train_model(self, observation, action, reward):
        """Train Random Forest approximator"""
        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array(reward).reshape((1, 1))
        if self.x_train is not None:
            self.x_train = np.append(self.x_train, x, axis=1)
        else:
            self.x_train = x
        if self.reward_theta is not None:
            self.reward_theta = np.append(self.reward_theta, y, axis=0)
        else:
            self.reward_theta = y   

        #re-train with increasing time interval, otherwise only update leaf values
        if self.t >= self.initial_random_selections and np.ceil(8*np.log(self.t)) > self.retrain_check:

            self.retrain_check = np.ceil(8*np.log(self.t))  #set to e.g. 0 to re-train every step

            X = self.x_train.T
            y = self.reward_theta.flatten()
            
            #train RF model
            self.rf_model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=self.model_parameters['min_child_weight'])
            self.rf_model.fit(X, y)

            logging.info('t = ' + str(self.t) + ' - re-trained model')

            leaf_of_data = self.rf_model.apply(X)

            y_weighted = y / self.n_estimators
            self.leaves_per_tree = []

            for i in range(self.n_estimators):
                self.leaves_per_tree.append({})
                df = pd.DataFrame({'leaf':leaf_of_data[:,i], 'weighted_target':y_weighted})

                group = df.groupby(by='leaf').agg(['count', 'mean', 'var'])
                for leaf_idx, row in group.iterrows():
                    self.leaves_per_tree[i][str(leaf_idx)] = dict()
                    self.leaves_per_tree[i][str(leaf_idx)]["leaf_count"] = row['weighted_target']['count']
                    self.leaves_per_tree[i][str(leaf_idx)]["leaf_mean"] = row['weighted_target']['mean']
                    self.leaves_per_tree[i][str(leaf_idx)]["leaf_variance"] = row['weighted_target']['var']

        elif self.t >= self.initial_random_selections:
            X = x.T
            y_weighted = y.flatten() / self.n_estimators

            leaf_preds = self.rf_model.apply(X)

            for i in range(self.n_estimators):
                self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_mean"] = ( (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_mean"] * 
                                                                        self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] + 
                                                                        y_weighted) / (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] + 1) )[0]
                self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_variance"] = ( (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_variance"] *
                                                                            (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] - 1) + 
                                                                            (y_weighted - self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_mean"])**2 ) / 
                                                                            (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"]) )[0]
                self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] += 1
        
        return

    def predict(self, x):
        """Predict reward"""

        if self.t > self.initial_random_selections:
            X = x.T
            
            leaf_assignments = self.rf_model.apply(X)
            
            n_samples_total = len(self.reward_theta)
        
            mean_per_assigned_leaf = np.array([self.leaves_per_tree[i][str(leaf_assignments[0][i])]["leaf_mean"] for i in range(len(self.leaves_per_tree))])
            mu_hat = np.sum(mean_per_assigned_leaf)
     
            variance_of_mean_per_assigned_leaf = np.array([self.leaves_per_tree[i][str(leaf_assignments[0][i])]["leaf_variance"] / 
            self.leaves_per_tree[i][str(leaf_assignments[0][i])]["leaf_count"] for i in range(len(self.leaves_per_tree))])         
            total_variance_of_mean = np.sum(variance_of_mean_per_assigned_leaf)

            c = self.exploration_factor * np.sqrt(total_variance_of_mean * np.log(self.t - 1) / n_samples_total)
        
        else:
            mu_hat = 0.0
            c = 0.0

        return mu_hat + c

    def get_samples(self, observation, t):
        samples = dict()
        for arm in self.internal_env.arm_set:
            x = observation[arm].context
            pred = self.predict(x)
            samples[arm] = pred

        return samples


    def update_observation(self, observation, action, reward):
        self.train_model(observation, action, reward)

        return

    def pick_action(self, observation):

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
        else:
            samples = self.get_samples(observation, self.t)
            self.internal_env.overwrite_arm_weight(samples)
            arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1

        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data


class BernoulliRandomForestTETS(Agent):
    """TETS with Random Forest for Bernoulli rewards"""

    def __init__(self, env_constructor, context_size, 
                use_cuda=False, exploration_variance=1.0, max_depth=6,
                n_estimators=100, initial_random_selections=10):

        self.t = 1
        self.context_size = context_size

        self.internal_env = env_constructor()

        self.exploration_variance = exploration_variance
        self.use_cuda = use_cuda
        self.x_train = None
        self.reward_theta = None

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.max_depth = max_depth
        self.n_estimators = n_estimators

        self.initial_random_selections = initial_random_selections
        self.retrain_check = 0

        print('Random Forest parameters - exploration_variance:', self.exploration_variance, 'n_estimators:', self.n_estimators)
        
        self.model_parameters = {'min_child_weight': 2}
        self.iteration_data = dict()
        Agent.__init__(self)

    def train_model(self, observation, action, reward):
        """Train Random Forest approximator"""
        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array(reward).reshape((1, 1))
        if self.x_train is not None:
            self.x_train = np.append(self.x_train, x, axis=1)
        else:
            self.x_train = x
        if self.reward_theta is not None:
            self.reward_theta = np.append(self.reward_theta, y, axis=0)
        else:
            self.reward_theta = y 

        #re-train with increasing time interval, otherwise only update leaf values
        if self.t >= self.initial_random_selections and np.ceil(8*np.log(self.t)) > self.retrain_check:
        
            self.retrain_check = np.ceil(8*np.log(self.t))  #set to e.g. 0 to re-train every step

            X = self.x_train.T
            y = self.reward_theta.flatten()
            
            #train RF model
            self.rf_model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=self.model_parameters['min_child_weight'])
            self.rf_model.fit(X, y)

            logging.info('t = ' + str(self.t) + ' - re-trained model')

            leaf_of_data = self.rf_model.apply(X)

            y_weighted = y / self.n_estimators
            self.leaves_per_tree = []

            for i in range(self.n_estimators):
                self.leaves_per_tree.append({})
                df = pd.DataFrame({'leaf':leaf_of_data[:,i], 'weighted_target':y_weighted})

                group = df.groupby(by='leaf').agg(['count', 'mean', 'var'])
                for leaf_idx, row in group.iterrows():
                    self.leaves_per_tree[i][str(leaf_idx)] = dict()
                    self.leaves_per_tree[i][str(leaf_idx)]["leaf_count"] = row['weighted_target']['count']
                    self.leaves_per_tree[i][str(leaf_idx)]["leaf_mean"] = row['weighted_target']['mean']
                    self.leaves_per_tree[i][str(leaf_idx)]["leaf_variance"] = row['weighted_target']['var']

        elif self.t >= self.initial_random_selections:
            X = x.T
            y_weighted = y.flatten() / self.n_estimators

            leaf_preds = self.rf_model.apply(X)

            for i in range(self.n_estimators):
                self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_mean"] = ( (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_mean"] * 
                                                                        self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] + 
                                                                        y_weighted) / (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] + 1) )[0]
                self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_variance"] = ( (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_variance"] *
                                                                            (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] - 1) + 
                                                                            (y_weighted - self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_mean"])**2 ) / 
                                                                            (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"]) )[0]
                self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] += 1
        
        return


    def predict(self, x):
        """Predict reward"""

        if self.t > self.initial_random_selections:
            X = x.T
            
            leaf_assignments = self.rf_model.apply(X)
        
            mean_per_assigned_leaf = np.array([self.leaves_per_tree[i][str(leaf_assignments[0][i])]["leaf_mean"] for i in range(len(self.leaves_per_tree))])

            mu_hat = np.sum(mean_per_assigned_leaf)

            variance_of_mean_per_assigned_leaf = np.array([self.leaves_per_tree[i][str(leaf_assignments[0][i])]["leaf_variance"] / 
            self.leaves_per_tree[i][str(leaf_assignments[0][i])]["leaf_count"] for i in range(len(self.leaves_per_tree))])         
            total_variance_of_mean = np.sum(variance_of_mean_per_assigned_leaf)

        else:
            mu_hat = 0.0
            total_variance_of_mean = self.exploration_variance  #dummy value

        return mu_hat + np.sqrt(total_variance_of_mean) * np.random.randn()

    def get_samples(self, observation, t):
        samples = dict()
        for arm in self.internal_env.arm_set:
            x = observation[arm].context
            pred = self.predict(x)
            samples[arm] = pred
        return samples


    def update_observation(self, observation, action, reward):
        self.train_model(observation, action, reward)

        return

    def pick_action(self, observation):

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
        else:
            samples = self.get_samples(observation, self.t)
            self.internal_env.overwrite_arm_weight(samples)
            arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1

        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data


class BernoulliTreeBootstrapRandomForest(Agent):
    """TreeBootstrap with Random Forest for Bernoulli rewards"""

    def __init__(self, env_constructor,
                 context_size, use_cuda=True,
                 initial_random_selections=10):

        self.t = 1
        self.context_size = context_size
        self.iteration_data = dict()

        self.internal_env = env_constructor()

        self.initial_random_selections = initial_random_selections
        self.use_cuda = use_cuda

        self.D = defaultdict()
        self.rewards = defaultdict()
        for arm in self.internal_env.arm_set:
                self.D[arm] = None
                self.rewards[arm] = None

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        Agent.__init__(self)

    
    def get_sampled_rewards(self, observation):

        sampled_mean_rewards = defaultdict()

        temp_list = []
        logging.debug("observation", observation)
        for arm in observation:
            logging.debug("observation[arm]",observation[arm])
            x = observation[arm].context.reshape((self.context_size, 1))

            bootstrap_indices = np.random.choice(range(self.D[arm].shape[1]), self.D[arm].shape[1], replace=True)

            D_bar = self.D[arm].T[bootstrap_indices]
            rewards_bar = self.rewards[arm][bootstrap_indices].flatten()

            predictor = RandomForestRegressor(n_estimators=100, max_depth=10)
            predictor.fit(D_bar, rewards_bar)
            mean_reward = predictor.predict(x.T)
            
            sampled_mean_rewards[arm] = mean_reward

        return sampled_mean_rewards
        

    def update_observation(self, observation, action, reward):

        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array(reward).reshape((1, 1))
        if self.D[action.label] is not None:
            self.D[action.label] = np.append(self.D[action.label], x, axis=1)
            self.rewards[action.label] = np.append(self.rewards[action.label], y, axis=0)
        else:
            self.D[action.label] = x
            self.rewards[action.label] = y

        return


    def pick_action(self, observation):

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
        else:
            samples = self.get_sampled_rewards(observation)
            self.internal_env.overwrite_arm_weight(samples)
            arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1

        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data


class BernoulliTreeBootstrapXGBoost(Agent):
    """TreeBootstrap with XGBoost for Bernoulli rewards"""

    def __init__(self, env_constructor,
                 context_size, use_cuda=True,
                 enable_categorical=True,
                 initial_random_selections=10,
                 feature_types=None):

        self.t = 1
        self.context_size = context_size
        self.iteration_data = dict()

        self.internal_env = env_constructor()

        self.initial_random_selections = initial_random_selections
        self.use_cuda = use_cuda

        self.D = defaultdict()
        self.rewards = defaultdict()
        for arm in self.internal_env.arm_set:
                self.D[arm] = None
                self.rewards[arm] = None

        self.feature_types = feature_types
        self.enable_categorical = enable_categorical

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        Agent.__init__(self)

    
    def get_sampled_rewards(self, observation):

        sampled_mean_rewards = defaultdict()

        temp_list = []
        logging.debug("observation", observation)
        for arm in observation:
            logging.debug("observation[arm]",observation[arm])
            x = observation[arm].context.reshape((self.context_size, 1))

            bootstrap_indices = np.random.choice(range(self.D[arm].shape[1]), self.D[arm].shape[1], replace=True)
            D_bar = self.D[arm].T[bootstrap_indices]
            rewards_bar = self.rewards[arm][bootstrap_indices].flatten()

            model_parameters = {'booster': 'gbtree', 'tree_method': 'hist', 'objective': 'reg:squarederror', 
                                'max_depth': 10}

            Xy = xgb.DMatrix(D_bar, rewards_bar, feature_types=self.feature_types, enable_categorical=self.enable_categorical)
            X = xgb.DMatrix(x.T, feature_types=self.feature_types, enable_categorical=self.enable_categorical)
            model = xgb.train(params=model_parameters, dtrain=Xy, num_boost_round=100)
            mean_reward = model.predict(X)
            
            sampled_mean_rewards[arm] = mean_reward

        return sampled_mean_rewards

    def update_observation(self, observation, action, reward):

        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array(reward).reshape((1, 1))
        if self.D[action.label] is not None:
            self.D[action.label] = np.append(self.D[action.label], x, axis=1)
            self.rewards[action.label] = np.append(self.rewards[action.label], y, axis=0)
        else:
            self.D[action.label] = x
            self.rewards[action.label] = y

        return

    def pick_action(self, observation):

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
        else:
            samples = self.get_sampled_rewards(observation)
            self.internal_env.overwrite_arm_weight(samples)
            arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1

        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data


class BernoulliTreeBootstrapDecisionTree(Agent):
    """TreeBootstrap with single Decision Tree for Bernoulli rewards"""

    def __init__(self, env_constructor,
                 context_size, use_cuda=True,
                 initial_random_selections=10):

        self.t = 1
        self.context_size = context_size
        self.iteration_data = dict()

        self.internal_env = env_constructor()

        self.initial_random_selections = initial_random_selections
        self.use_cuda = use_cuda

        self.D = defaultdict()
        self.rewards = defaultdict()
        for arm in self.internal_env.arm_set:
                self.D[arm] = None
                self.rewards[arm] = None

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        Agent.__init__(self)

    
    def get_sampled_rewards(self, observation):

        sampled_mean_rewards = defaultdict()

        temp_list = []
        logging.debug("observation", observation)
        for arm in observation:
            logging.debug("observation[arm]",observation[arm])
            x = observation[arm].context.reshape((self.context_size, 1))
            bootstrap_indices = np.random.choice(range(self.D[arm].shape[1]), self.D[arm].shape[1], replace=True)
            D_bar = self.D[arm].T[bootstrap_indices]
            rewards_bar = self.rewards[arm][bootstrap_indices].flatten()

            predictor = DecisionTreeRegressor()
            predictor.fit(D_bar, rewards_bar)
            mean_reward = predictor.predict(x.T)
            
            sampled_mean_rewards[arm] = mean_reward

        return sampled_mean_rewards
        

    def update_observation(self, observation, action, reward):

        x = observation[action.label].context.reshape((self.context_size, 1))
        y = np.array(reward).reshape((1, 1))
        if self.D[action.label] is not None:
            self.D[action.label] = np.append(self.D[action.label], x, axis=1)
            self.rewards[action.label] = np.append(self.rewards[action.label], y, axis=0)
        else:
            self.D[action.label] = x
            self.rewards[action.label] = y

        return


    def pick_action(self, observation):

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
        else:
            samples = self.get_sampled_rewards(observation)
            self.internal_env.overwrite_arm_weight(samples)
            arm_selection = self.internal_env.get_optimal_action()
        self.t = self.t + 1

        return arm_selection.label

    def get_iteration_data(self):
        return self.iteration_data