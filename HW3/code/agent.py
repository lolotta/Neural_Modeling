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
        # puts the newest s,a,r,s1 experience in the experience buffer at the position s,a
        # and replaces old experience
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
        # updates the q value for a given state-action pair
        # the exploration bonus is only added during planning when bonus = true
        # and only added for dyna q+ which has epsilon != 0
        # in dyna q-, epsilon is 0 and therefore the exploration bonus is not added

        if self.epsilon != 1:
            self.Q[s,a] = self.Q[s,a] + self.alpha*(r + (self.epsilon*bonus*np.sqrt(self.action_count[s,a])) + self.gamma*next_Q - self.Q[s,a])
        else:
            self.Q[s,a] = self.Q[s,a] + self.alpha*(r + self.gamma*next_Q - self.Q[s,a])

        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''
        # keeps track of how many steps ago an action in a state was performed
        # after every step, all action counts are increased
        # only the action that was taken is reset to 0
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
        # adds s,a,r,s1 to the history
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
        #since epsilon is 0 in dyna q-, no exploration bonus is added there
        adapted_qs = qs + self.epsilon*np.sqrt(ns)

        #if there is multiple adapted qs with same value, make a random choice 
        # between them. Im using noise to make the values different
        # the noise doesnt change the q values itself 
        
        if self.epsilon != 1:
            if np.unique(adapted_qs).shape[0] < 4:
                adapted_qs += np.random.randn(4) * 0.000001

            a = np.argmax(np.asarray(adapted_qs))



        # this is for epsilon greedy agent, not for dyna:
        else:
            epsi = 0.1
            if np.random.random() < epsi:
                a = np.random.randint(0,4)
            else:
                if np.unique(qs).shape[0] < 4:
                    random_noise = np.random.randn(4) * 0.000001
                    qs += random_noise
                    a = np.argmax(np.asarray(qs))
                    qs -= random_noise
                else:
                    a = np.argmax(np.asarray(qs))

        return a

    def _plan(self, num_planning_updates):

        '''
        Planning computations
        Input arguments:
            num_planning_updates -- number of planning updates to execute
        '''

        # complete the code
        # for the number of planning steps, perform a random action from the
        # experience buffer again and update the q values with those experiences
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

        return None
        
    def init_env(self, env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''
        Environment_TwoStepAgent.__init__(self, env_config)
        return None
    
    
    def _init_q_values(self):

        '''
        Initialise the Q-value tables

        QTD = model free
        QMB = model based
        Qnet = combination of the two above
        '''

        self.QTD = np.zeros((self.num_states* self.num_actions))
        self.QMB = np.zeros((2)) #for the other s-a combinations, QMB is not needed and QTD is used
        self.Qnet = np.zeros((self.num_states* self.num_actions))
        return None
    

    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _init_reward(self):

        '''
        Initialise rewards uniformly and determine boundaries
        '''

        self.rewards = np.random.uniform(0.25,0.75,4)#[0.5,0.5,0.5,0.5]
        self.boundaries = [0.25,0.75]


        return None
    
    def get_reward(self, s,a):
        '''
        returns the reward at a given final state
        since final state is s=3 everywhere, the true
        final state is calculated as combination of s2 and a2
        '''
        
        #final_state 0: sB + left
        #final_state 1: sB + right
        #final_state 2: sC + left
        #final_state 3: sC + right
        # input s is 1 or 2
        # input a is 0 or 1
        position = (s-1)*2+a
        p = self.rewards[position]
        r = np.random.choice((0,1), p=(1-p, p))

        return r
    
    def update_rewards(self):
        '''
        changes rewards by a gaussian noise
        and keeps it in boundaries
        '''

        #add gaussian noise to every reward
        self.rewards += np.random.normal(loc=0, scale=0.025, size=4)
        #check if rewards are outside the boundaries and make the boundary reflective
        for i, reward in enumerate(self.rewards):
            if (reward < self.boundaries[0]):
                difference = self.boundaries[0] - reward
                self.rewards[i] += 2*difference

            elif (reward>self.boundaries[1]):
                difference = self.boundaries[1] - reward
                self.rewards[i] += 2*difference

    def _init_transition_matrix(self):

        '''
        Initialise transition matrix with transition possibilities and
        initialise current transition matrix that keeps track of current belief 

        '''

        self.transition_m = [[0.7,0.3], [0.3,0.7]] #[[p from going to sB with action 0, or with 1], [p from going to sB with action 0, or 1]]
        self.current_transition_m = self.transition_m[0] #set an initial believed transition

        return None
    
    def _init_track_state_transition(self):
        '''
        Initialise counters for evidence of either transition matrix and
        Initialise memory for last action taken at s=0 


        '''
        self.transition_1 = 0
        self.transition_2 = 0

        self.last_a = -1


    def _track_state_transitions(self, a, s1):
        '''
        Increase counter when evidence is brought for either transition matrix hypothesis

        '''

        if (s1==1 and a==0) or (s1==2 and a==1):
            self.transition_1+=1
        elif (s1==2 and a==0) or (s1==1 and a==1):
            self.transition_2+=1


    def _update_q_td(self, s, a, r, s1, a1, trace):

        '''
        Update the Q-value table of td
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            a1    -- next action
            trace -- if eligibility trace is applied
        '''
        
        
        if s == 0:
            s_a_old = s*self.num_actions+a
            s_a_new = s1*self.num_actions+a1
            q_s1 = self.QTD[s_a_new]
            alpha = self.alpha1

            if trace:
                # eligibility trace is applied by using reward of s=3 and multiplying by lambda
                delta = r-q_s1
                self.QTD[s_a_old] += alpha*delta*self.lam
            else:
                # normal q value update
                q_s = self.QTD[s_a_old]
                delta = r+q_s1-q_s
                self.QTD[s_a_old] += alpha*delta
        # else there is no next state
        else:
            s_a = s*self.num_actions+a
            q_s = self.QTD[s_a]
            alpha = self.alpha2
            delta = r-q_s
            self.QTD[s_a] += alpha*delta


        return None
    
    def _update_q_mb(self, s, a):

        '''
        Update the Q-value table of mb
        Input arguments:
            s     -- initial state
            a     -- chosen action
        '''

        # choose current transition matrix based on where more transitions happened so far
        if self.transition_1 > self.transition_2:
            self.current_transition_m = self.transition_m[0]
        else:
            self.current_transition_m = self.transition_m[1]

        s_a = s*self.num_actions+a

        # in s=0 update QMB based on the update rule from the paper
        prob1 = self.current_transition_m[a]
        prob2 = 1-prob1

        best_q1 = np.max(self.QTD[2:4]) 
        best_q2 = np.max(self.QTD[4:])
        self.QMB[s_a] = prob1*best_q1 +prob2*best_q2

        
        return None
    
    def _update_q_net(self, s, a):

        '''
        Update the Q-value table of combined td and mb
        Input arguments:
            s     -- state
            a     -- chosen action

        '''
        s_a = s*self.num_actions+a

        # in s=0, Qnet is a weighted sum of QMB and QTD (all at positions s,a) 
        self.Qnet[s_a] = self.w * self.QMB[s_a] + (1-self.w)*self.QTD[s_a]

        return None

    def _update_history(self, a, s1, r):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''
        self.history = np.vstack((self.history, np.array([a, s1, r])))

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
        # rep is 1 if we are in the first stage and the current action is the same
        # as in the last trial. rep is 0 else
        for a in range(self.num_actions):
            if s == 0:
                beta = self.beta1
                rep = self.last_a == a
            else:
                beta = self.beta2
                rep = 0
            s_a = s*self.num_actions+a

            exp_terms[a] = np.exp(beta*(self.Qnet[s_a] + self.p * rep)) 

        # softmax 
        policy =  exp_terms/ exp_terms.sum()

        a = np.random.choice(np.arange(2),p=policy)
        return a
    
    def get_next_state(self,s,a):
        if s == 0:
            # first transition matrix is the true one
            prob1 = self.transition_m[0][a]
            #prob1 is the probability of going to sB with the chosen action
            p = [prob1, 1-prob1]
            p=np.round(p,1)
            new_s = np.random.choice([1,2], p=p)
        else:
            new_s = 3
        return new_s

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
        # initialise everything
        self._init_q_values()
        self._init_track_state_transition()
        self._init_history()
        self._init_transition_matrix()
        self._init_reward()

        for _ in range(num_trials):
            # start at s=0
            s1 = self.start_state
            # choose action
            a1  = self._policy(s1)
            # reward 1 is always 0
            r1=0
            # get new state
            s2 = self.get_next_state(s1,a1)

            # track the transition that occured from stage 1 to stage 2 and memorise the action 
            self._track_state_transitions(a1,s2)
            self.last_a = a1

            # choose second action
            a2 = self._policy(s2)

            # get the probabilistic reward
            r2 = self.get_reward(s2,a2)

            # learning
            # update model free q value
            self._update_q_td(s1, a1, r1, s2,a2,False)
            self._update_q_td(s2, a2, r2, _,_,False)
            # update model free q value with eligibility trace
            self._update_q_td(s1, a1, r2, s2,a2,True)
            # update model based q values for BOTH actions
            self._update_q_mb(s1,a1)
            self._update_q_mb(s1,1-a1)
 
            #update all q net values
            self._update_q_net(s1,a1)
            self._update_q_net(s1,1-a1)
            self.Qnet[2:] = self.QTD[2:]

            #update history
            self._update_history(a1, s2, r2)
            # add gaussian noise to rewards
            self.update_rewards()
            
        return None