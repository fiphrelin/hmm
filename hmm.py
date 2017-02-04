import numpy as np
import pandas as pd

   
class HMM:
    def __init__(self, A, B, p0=None):
        self.A = A
        self.B = B
        n = A.shape[0]
        self.p0 = np.ones(n) / float(n)

    def copy(self):
        return HMM(self.A.copy(),
                   self.B.copy(),
                   self.p0)
    
    def forward_(self, x, t, l):
        if t==0:
            l.append(self.p0 * self.B.loc[:, x[t]].values[np.newaxis, :])
        else:
            l.append(self.forward_(x, t-1, l).dot(self.A.values) * self.B.loc[:, x[t]].values.T)
        return l[-1]

    def forward(self, x, return_proba=True):
        '''
        Forward algorithm (computing the likelihood of an observation sequence).

        Parameters
        ----------
        x: list (TODO check __iter__)
             observation sequence (sequence of discrete emissions)

        t: int, default=len(x)-1
             which index must the forward procedure be stopped at (included)
        
        keep_history: bool, default=False
             whether or not to keep all the intermediary calculations.

        Returns
        -------
        lk: float if not keep_history else list
             if not keep_history, lk of the sequence up until time t
             otherwise likelihood of each class at each time up until time t

        Examples
        --------
        >>> hidden_states = list('ABC')
        >>> emissions = list('xyz')
        >>> n = len(hidden_states)
        >>> p = len(emissions)
        >>> A = np.array([ 0.555,  0.191,  0.254,  0.037,  0.619,  0.344,  0.19 ,  0.575, 0.234]).reshape((n, n))
        >>> A = pd.DataFrame(A, index=hidden_states, columns=hidden_states)
        >>> B = np.array([ 0.133,  0.226,  0.641,  0.27 ,  0.718,  0.013,  0.348,  0.222, 0.43 ]).reshape((n, p))
        >>> B = pd.DataFrame(B, index=hidden_states, columns=emissions)
        >>> hmm = HMM(A, B)
        >>> x = 'xzzxxzzyxyxzyxyyzxxyxxzzyxzxxzxxxyxxzyyyyxyxxzxxzyzzyzxyyxyzzxzzyyxxyyyyyxyyxyxxxzyxyyyzxzyxyzzyxyzy'
        >>> - np.log(hmm.forward(x))
        112.66967033011052
        '''
        l = list()
        self.forward_(x, len(x)-1, l)
        l = np.array(l)
        if return_proba:
            return l[-1].sum()
        else:
            return l

    def backward_(self, x, t, l):
        if t==len(x)-1:
            l.append(np.ones((1, self.A.shape[0])))
        else:
            l.append((self.backward_(x, t+1, l) * self.B.loc[:, x[t+1]].values).dot(self.A.values.T))
        return l[-1]

    def backward(self, x, return_proba=True):
        '''
        TODO

        Parameters
        ----------
        x: list (TODO check __iter__)
             observation sequence (sequence of discrete emissions)

        t: int, default=len(x)-1
             which index must the forward procedure be stopped at (included)
        
        keep_history: bool, default=False
             whether or not to keep all the intermediary calculations.

        Returns
        -------
        lk: float if not keep_history else list
             if not keep_history, lk of the sequence up until time t
             otherwise likelihood of each class at each time up until time t
        '''
        
        l = list()
        self.backward_(x, 0, l)
        l = np.array(l)[::-1]
        if return_proba:
            return l[-1].sum()
        else:
            return l
                        
    def viterbi_(self, x, t, state_seq):
        '''

        '''
        if t==0:
            return (self.p0 * self.B.loc[:, x[t]].values)
        else:
            p_hidden_t = self.viterbi_(x, t-1, state_seq)[:, np.newaxis] * self.A.values
            state_seq.append(np.argmax(p_hidden_t, axis=0))
            return (p_hidden_t.max(axis=0) * self.B.loc[:, x[t]].values)
      
    def viterbi(self, x):
        '''
        TODO

        Parameters
        ----------
        x: list (TODO check __iter__)
             observation sequence (sequence of discrete emissions)

        t: int, default=len(x)-1
             which index must the forward procedure be stopped at (included)
        
        Returns
        -------
        s = list
            list of hidden states

        Examples
        --------
        >>> hidden_states = list('ABC')
        >>> emissions = list('xyz')
        >>> n = len(hidden_states)
        >>> p = len(emissions)
        >>> A = np.array([ 0.466,  0.378,  0.156,  0.149,  0.309,  0.542,  0.241,  0.448, 0.311]).reshape((n, n))
        >>> A = pd.DataFrame(A, index=hidden_states, columns=hidden_states)
        >>> B = np.array([ 0.231,  0.708,  0.061,  0.129,  0.304,  0.567,  0.079,  0.288, 0.633]).reshape((n, p))
        >>> B = pd.DataFrame(B, index=hidden_states, columns=emissions)
        >>> hmm = HMM(A, B)
        >>> x = 'yxyzyyxyxxyyyxzzzyxxxxzzyyzxzzxyyxxxyyzxxxyxzyzzzyyyyxzzyyyyzyzyyxxxxyyyyzxzxyzzzzxyxyzxxyxxxyzxzyxy'
        >>> ''.join(hmm.viterbi(x))
        'AAABAAAAAAAAAABCCAAAAABCAABCBCAAAAAAAABAAAABCBCBCAAAAABCAAAABBCAAAAAAAAAABBCAABCBCAAAABAAAAAAABBCAAA'
        '''
        state_seq = list() 
        d = self.viterbi_(x, len(x)-1, state_seq)
        
        # backtrack
        q = np.argmax(d)
        res = [q]
        for ti in range(len(x)-1)[::-1]:
            nq = state_seq[ti][q]
            res.append(nq)
            q = nq
        
        return self.A.index[res[::-1]].tolist()
    
    def soft_decoding(self, x):
        '''
        TODO

        Parameters
        ----------
        x: list (TODO check __iter__)
             observation sequence (sequence of discrete emissions)

        t: int, default=len(x)-1
             index for which to compute the probability of each classes
        
        Returns
        -------
        gamma_t: np.array, shape: (n_hidden_states, 1)
             gamma
        
        Examples
        --------
        >>> hidden_states = list('AB')
        >>> emissions = list('xyz')
        >>> n = len(hidden_states)
        >>> p = len(emissions)
        >>> A = np.array([ 0.095,  0.905,  0.225,  0.775]).reshape((n, n))
        >>> A = pd.DataFrame(A, index=hidden_states, columns=hidden_states)
        >>> B = np.array([ 0.491,  0.311,  0.198,  0.47 ,  0.495,  0.036]).reshape((n, p))
        >>> B = pd.DataFrame(B, index=hidden_states, columns=emissions)
        >>> hmm = HMM(A, B)
        >>> x = 'xxxyzyyzxz'
        >>> hmm.soft_decoding(x)[-1]
        array([[ 0.58641854,  0.41358146]])
        '''

        alpha = self.forward(x, return_proba=False)
        beta = self.backward(x, return_proba=False)
        ab = alpha * beta
        return ab / ab.sum(axis=-1)[:, np.newaxis]
    
    def viterbi_learning_(self, x):
        '''
        Parameters
        ----------
        x: list (TODO check __iter__)
             observation sequence (sequence of discrete emissions)

        Examples
        --------
        >>> hidden_states = list('AB')
        >>> emissions = list('xyz')
        >>> n = len(hidden_states)
        >>> p = len(emissions)
        >>> A = np.array([ 0.609,  0.391,  0.461,  0.539]).reshape((n, n))
        >>> A = pd.DataFrame(A, index=hidden_states, columns=hidden_states)
        >>> B = np.array([ 0.473,  0.164,  0.363,  0.524,  0.01 ,  0.467]).reshape((n, p))
        >>> B = pd.DataFrame(B, index=hidden_states, columns=emissions)
        >>> hmm = HMM(A, B)
        >>> x = 'xzyzzyyxzzzzyyzzxxxzzyyyxzzzxyzzzxzyxyzzyzxyyxzzyyzzxxzxyzyzyzzxxzyyxxzxyxzzxzxxzxxzxyzyxyyxzyzyyyzy'
        >>> hmm.viterbi_learning_(x)
        >>> hmm.A
             A    B
        A  1.0  0.0
        B  0.5  0.5
        >>> hmm.B
                  x         y         z
        A  0.280000  0.310000  0.410000
        B  0.333333  0.333333  0.333333
        '''

        seq = self.viterbi(x)
        B = pd.DataFrame(np.zeros(self.B.shape),
                 index=self.B.index,
                 columns=self.B.columns)

        for i, j in zip(seq, x):
            B.loc[i, j]+=1

        nB = B.values / B.sum(axis=1).values[:, np.newaxis]
        nB[np.isnan(nB)] = 1./ B.shape[1]
        self.B = pd.DataFrame(nB, index=B.index, columns=B.columns)

        A = pd.DataFrame(np.zeros(self.A.shape),
                 index=self.A.index,
                 columns=self.A.index)
                 
        for i, j in zip(seq[:-1], seq[1:]):
            A.loc[i, j]+=1

        nA = A.values / A.sum(axis=1).values[:, np.newaxis]
        nA[np.isnan(nA)] = 1./A.shape[1]
        self.A = pd.DataFrame(nA, index=A.index, columns=A.columns)

    def baumwelch_learning_(self, x, n_iter=None, update_pi=True):
        '''
        baumwelch learning

        Parameters
        ----------
        x: list (TODO check __iter__)
             observation sequence (sequence of discrete emissions)

        n_iter: int
             number of EM steps
       
        update_pi: bool
             whether to update the distribution of the hidden_states or not
             with each EM step

        Examples
        --------
        >>> hidden_states = list('ABC')
        >>> emissions = list('xyz')
        >>> n = len(hidden_states)
        >>> p = len(emissions)
        >>> A = np.array([ 0.048,  0.449,  0.503,  0.323,  0.361,  0.316,  0.261,  0.406, 0.333]).reshape((n, n))
        >>> A = pd.DataFrame(A, index=hidden_states, columns=hidden_states)
        >>> B = np.array([ 0.209,  0.667,  0.124,  0.63 ,  0.171,  0.199,  0.334,  0.166,  0.5  ]).reshape((n, p))
        >>> B = pd.DataFrame(B, index=hidden_states, columns=emissions)
        >>> hmm = HMM(A, B)
        >>> x = 'xzzzyzzzzzyxyxxzxzxzxzzxyyyzzzxyzxxxxyxxzyxyxzxzzyyyxxyxxyyyyzxxxxyyzxyzzyzxxxxyyxyxzxxzxxxzzyyyyxxx'
        >>> n_iter = 100
        >>> hmm.baumwelch_learning_(x, n_iter, False)
        >>> hmm.A
                  A         B         C
        A  0.010351  0.238835  0.750814
        B  0.145034  0.671979  0.182987
        C  0.166824  0.202034  0.631142

        >>> hmm.B
                  x         y         z
        A  0.569151  0.408820  0.022029
        B  0.444220  0.552735  0.003044
        C  0.334909  0.041402  0.623688
        '''
        best_lk = 0.
        step = 0
        while (n_iter is None) or (step < n_iter):
            step += 1
            ksi = []
            alpha = self.forward(x, return_proba=False)
            beta = self.backward(x, return_proba=False)
            for t in range(len(x)-1):
                ksi.append(
                    alpha[t].T * 
                    self.A.values *
                    self.B.loc[:, x[t+1]].values[np.newaxis, :] *
                    beta[t+1]
                          )
            
            ksi = [k / k.sum() for k in ksi]
            gamma = self.soft_decoding(x)
            nA = sum(ksi) / gamma[:-1].sum(axis=0).T
            
            tmp = [gamma[np.array(list(x)) == emission].sum(axis=0)
                   for emission in self.B.columns]
            nB = np.array(tmp).reshape(self.B.T.shape).T / gamma.sum(axis=0).T
            
            # save current model
            copy = self.copy()

            if update_pi:
                self.p0 = gamma[0].flatten()
            self.A = pd.DataFrame(nA, index=self.A.index, columns=self.A.columns)
            self.B = pd.DataFrame(nB, index=self.B.index, columns=self.B.columns)
            
            lk = self.forward(x)
            if  lk <= best_lk:
                self = copy
                break
            else:
                best_lk = lk

    def fit(self, x, learning='baumwelch', **kwargs):
        '''
        Examples
        --------
        >>> hidden_states = list('ABC')
        >>> emissions = list('xyz')
        >>> n = len(hidden_states)
        >>> p = len(emissions)
        >>> A = np.array([ 0.048,  0.449,  0.503,  0.323,  0.361,  0.316,  0.261,  0.406, 0.333]).reshape((n, n))
        >>> A = pd.DataFrame(A, index=hidden_states, columns=hidden_states)
        >>> B = np.array([ 0.209,  0.667,  0.124,  0.63 ,  0.171,  0.199,  0.334,  0.166,  0.5  ]).reshape((n, p))
        >>> B = pd.DataFrame(B, index=hidden_states, columns=emissions)
        >>> hmm = HMM(A, B)
        >>> x = 'xzzzyzzzzzyxyxxzxzxzxzzxyyyzzzxyzxxxxyxxzyxyxzxzzyyyxxyxxyyyyzxxxxyyzxyzzyzxxxxyyxyxzxxzxxxzzyyyyxxx'
        >>> n_iter = 100
        >>> hmm.fit(x, n_iter=n_iter, update_pi=False)
        >>> hmm.fit(x, learning='viterbi')
        '''
        
        if learning == 'baumwelch':
            self.baumwelch_learning_(x, **kwargs)
        elif learning == 'viterbi':
            self.viterbi_learning_(x)
        else:
            raise NotImplementedError('Only "viterbi" and "baumwelch" learning procedures are implemented.')

if __name__ == '__main__':
    import doctest
    doctest.testmod()
