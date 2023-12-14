import numpy as np
from scipy import stats # for gaussian noise
from environment import Environment, Environment_TwoStepAgent

class DynaAgent(Environment):

    def __init__(self, alpha, gamma, epsilon):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha   -- learning rate \in (0, 1]
            gamma   -- discount factor \in (0, 1)
            epsilon -- controls the influence of the exploration bonus
        '''

        self.alpha   = alpha
        self.gamma   = gamma 
        self.epsilon = epsilon
        return None

    def init_env(self, env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''
        Environment.__init__(self, env_config)
        return None

    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.Q = np.zeros((self.num_states, self.num_actions))

        return None

    def _init_experience_buffer(self):

        '''
        Initialise the experience buffer
        '''

        self.experience_buffer = np.zeros((self.num_states*self.num_actions, 4), dtype=int)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.experience_buffer[s*self.num_actions+a] = [s, a, 0, s]

        return None

    def _init_history(self):

        '''
        Initialise the history
        '''

        self.history = np.empty((0, 4), dtype=int)
        # print(self.history)


        return None
    
    def _init_action_count(self):

        '''
        Initialise the action count
        '''

        self.action_count = np.zeros((self.num_states, self.num_actions), dtype=int)

        return None

    def _update_experience_buffer(self, s, a, r, s1):

        '''
        Update the experience buffer (world model)
        Input arguments:
            s  -- initial state
            a  -- chosen action
            r  -- received reward
            s1 -- next state
        '''
  
        self.experience_buffer[s*self.num_actions+a,:] = np.asarray((s,a,r,s1))
        return None

    def _update_qvals(self, s, a, r, s1, bonus=False):

        '''
        Update the Q-value table
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''
        best_a  = np.argmax(self.Q[s1,:])
        next_Q = self.Q[s1,best_a]

        # complete the code

        self.Q[s,a] = self.Q[s,a] + self.alpha*(r + (self.epsilon*bonus*np.sqrt(self.action_count[s,a])) + self.gamma*next_Q - self.Q[s,a])

        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''

        self.action_count = self.action_count + 1
        self.action_count[s,a] = 0
        return None

    def _update_history(self, s, a, r, s1):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''

        self.history = np.vstack((self.history, np.array([s, a, r, s1])))

        return None

    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''
        qs = self.Q[s,:]
        ns = self.action_count[s,:]
        adapted_qs = qs + 0.001*np.sqrt(ns)
        a = np.argmax(np.asarray(adapted_qs))

        # a = np.argmax(self.Q[s,:])
        
        # qs = self.Q[s,:]
        # probs = np.exp(qs)/np.exp(qs).sum()
        # a = np.random.choice(np.arange(4),p=probs)
        return a

    def _plan(self, num_planning_updates):

        '''
        Planning computations
        Input arguments:
            num_planning_updates -- number of planning updates to execute
        '''

        # complete the code
        for _ in range(num_planning_updates):

            sars_ind = np.random.randint(self.experience_buffer.shape[0])
            s,a,r,s1 = self.experience_buffer[sars_ind]
            
            self._update_qvals(s, a, r, s1, bonus=True)

        return None

    def get_performace(self):

        '''
        Returns cumulative reward collected prior to each move
        '''

        return np.cumsum(self.history[:, 2])

    def simulate(self, num_trials, reset_agent=True, num_planning_updates=None):

        '''
        Main simulation function
        Input arguments:
            num_trials           -- number of trials (i.e., moves) to simulate
            reset_agent          -- whether to reset all knowledge and begin at the start state
            num_planning_updates -- number of planning updates to execute after every move
        '''
        if reset_agent:

            self._init_q_values()
            self._init_experience_buffer()
            self._init_action_count()
            self._init_history()

            self.s = self.start_state

        for _ in range(num_trials):

            # choose action
            a  = self._policy(self.s)
            # get new state
            s1 = np.random.choice(np.arange(self.num_states), p=list(self.T[self.s, a, :]))
            # receive reward
            r  = self.R[self.s, a]
            # learning
            self._update_qvals(self.s, a, r, s1, bonus=False)
            # update world model 
            self._update_experience_buffer(self.s, a, r, s1)
            # reset action count
            self._update_action_count(self.s, a)
            # update history
            self._update_history(self.s, a, r, s1)
            # plan
            if num_planning_updates is not None:
                self._plan(num_planning_updates)

            if s1 == self.goal_state:
                self.s = self.start_state
            else:
                self.s = s1

        return None
    
class TwoStepAgent(Environment_TwoStepAgent):

    def __init__(self, alpha1, alpha2, beta1, beta2, lam, w, p):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha1 -- learning rate for the first stage \in (0, 1]
            alpha2 -- learning rate for the second stage \in (0, 1]
            beta1  -- inverse temperature for the first stage
            beta2  -- inverse temperature for the second stage
            lam    -- eligibility trace parameter
            w      -- mixing weight for MF vs MB \in [0, 1] 
            p      -- perseveration strength
        '''

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1  = beta1
        self.beta2  = beta2
        self.lam    = lam
        self.w      = w
        self.p      = p
        self.last_a = -1
        return None
        
    def init_env(self, env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''
        Environment.__init__(self, env_config)
        return None
    
    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.QTD = np.zeros(self.num_states* self.num_actions+1)
        self.QMB = np.zeros(self.num_states* self.num_actions+1)
        self.Qnet = np.zeros(self.num_states* self.num_actions+1)

        return None
    
    def _init_experience_buffer(self):

        '''
        Initialise the experience buffer
        '''

        self.experience_buffer = np.zeros((self.num_states*self.num_actions, 5), dtype=int)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.experience_buffer[s*self.num_actions+a] = [s, a, 0, s, a]

        return None

    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _init_transition_probabilities(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.transition_p = np.zeros(4)

        return None
    
    def _update_experience_buffer(self, s, a, r, s1, a1):

        '''
        Update the experience buffer (world model)
        Input arguments:
            s  -- initial state
            a  -- chosen action
            r  -- received reward
            s1 -- next state
        '''
        new_exp = np.asarray((s,a,r,s1,a1))
        if self.experience_buffer[s*self.num_actions+a].shape[0] == 1:
            self.experience_buffer[s*self.num_actions+a] == new_exp
        else:
            old_exp = self.experience_buffer[s*self.num_actions+a]
            self.experience_buffer[s*self.num_actions+a] = np.concatenate((old_exp, new_exp))
  
        return None
    
    def _get_transition_probabilities(self):
        self.transition_p = np.zeros(4)
        #p=0: prob(s2|s1,a1) 
        #p=1: prob(s2|s1,a2) 
        #p=2: prob(s3|s1,a1)  = 1 -p=0
        #p=3: prob(s3|s1,a2) = 1 -p=1
        for s in range(1):
            for a in range(2):
                experience = self.experience_buffer[0,a]
                counter = 0
                for exp in experience:
                    if exp[3] == s+1:
                        counter+=1
                self.transition_p[s+a] = counter/experience.shape[0]
                self.transition_p[s+a+2] = 1- self.transition_p[s+a]
        return None

    def _update_q_td(self, s, a, r, s1, a1):

        '''
        Update the Q-value table of td
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''
        delta = r + self.QTD[s1*self.num_actions+a1] - self.QTD[s*self.num_actions+a]
        self.QTD[s*self.num_actions+a] += self.alpha1*self.lam*delta
        return None
    
    def _update_q_mb(self, s, a, r, s1, a1):

        '''
        Update the Q-value table of mb
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''
        s_a = s*self.num_actions+a

        if s == 0:
            prob1 = self.transition_p[(s1-1)*self.num_actions+a]
            prob2 = 1-prob1
            best_q1 = np.max(self.QTD[1*self.num_actions:s1*self.num_actions+self.num_actions])
            best_q2 = np.max(self.QTD[2*self.num_actions:s1*self.num_actions+self.num_actions])
            self.QMB[s_a] = prob1*best_q1 +prob2*best_q2
        else:
            self.QMB[s_a] = self.QTD[s_a]
        
        return None
    
    def _update_q_net(self, s, a, r, s1, a1):

        '''
        Update the Q-value table of combined td and mb
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''
        s_a = s*self.num_actions+a

        if s == 0:
            self.Qnet[s_a] = self.w * self.QMB[s_a] + (1-self.w)*self.QTD[s_a]
        else:
            self.Qnet[s_a] = self.QTD[s_a]

        return None

    def _update_history(self, a, s1, r1):

        '''
        Update history
        Input arguments:
            a  -- first stage action
            s1 -- second stage state
            r1 -- second stage reward
        '''

        self.history = np.vstack((self.history, [a, s1, r1]))

        return None
    
    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''
        exp_terms = np.zeros(2)
        if s == 0:
            beta = self.beta1
            rep = self.last_a == a
            self.last_a = a
        else:
            beta = self.beta2
            rep = 0
        for a in self.num_actions:
            s_a = s*self.num_actions+a

            exp_terms[a] = np.exp(beta*(self.Qnet[s_a] + self.p * rep))

        policy =  exp_terms/ exp_terms.sum()

        a = np.random.choice(np.arange(2),p=policy)
        return a
    
    def get_stay_probabilities(self):

        '''
        Calculate stay probabilities
        '''

        common_r      = 0
        num_common_r  = 0
        common_nr     = 0
        num_common_nr = 0
        rare_r        = 0
        num_rare_r    = 0
        rare_nr       = 0
        num_rare_nr   = 0

        num_trials = self.history.shape[0]
        for idx_trial in range(num_trials-1):
            a, s1, r1 = self.history[idx_trial, :]
            a_next    = self.history[idx_trial+1, 0]

            # common
            if (a == 0 and s1 == 1) or (a == 1 and s1 == 2):
                # rewarded
                if r1 == 1:
                    if a == a_next:
                        common_r += 1
                    num_common_r += 1
                else:
                    if a == a_next:
                        common_nr += 1
                    num_common_nr += 1
            else:
                if r1 == 1:
                    if a == a_next:
                        rare_r += 1
                    num_rare_r += 1
                else:
                    if a == a_next:
                        rare_nr += 1
                    num_rare_nr += 1

        return np.array([common_r/num_common_r, rare_r/num_rare_r, common_nr/num_common_nr, rare_nr/num_rare_nr])

    def simulate(self, num_trials):

        '''
        Main simulation function
        Input arguments:
            num_trials -- number of trials to simulate
        '''
            

        self._init_q_values()
        self._init_experience_buffer()
        self._init_history()
        self._init_transition_probabilities

        self.s = self.start_state

        for _ in range(num_trials):
            self.s = self.start_state

            for stage in range(2):
                # choose action
                a  = self._policy(self.s)
                # get new state
                new_s = np.random.choice(np.arange(self.num_states), p=list(self.T[self.s, a, :]))
                # receive reward
                r  = self.R[self.s, a]
                # learning
                self._update_qvals(self.s, a, r, new_s)
                # update world model 
                self._update_experience_buffer(self.s, a, r, new_s)
                # update history
                self._update_history(self.s, a, r, new_s)
                
                s = new_s

        return None