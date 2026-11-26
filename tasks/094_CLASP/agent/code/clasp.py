"""CLASP-I from official notebook cell 5; causal data entry only."""
import numpy as np
import cvxpy as cp
n, k, T = 10, 4, 100
F, G, D = n**2, np.sqrt(4*n), np.sqrt(2)

class CLASP_I:
    
    def __init__(self, x_0, T, assumptions_param):
        
        self.x_0 = x_0        # Initial guess
        self.n = x_0.shape[0] # Optmization variable in R^n
        
        
    def projection_step(self, data):
        
        x, A, b = data
        
        y = cp.Variable((n,1))
        cost = (1/2) * cp.norm(y - x)**2

        constraints = [y <= 1, y >= 0]
        constraints.append(A @ y <= b)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        return y.value.copy()
    
    def run(self, episode):
        
        # Start with initial guess
        x_pred = self.x_0.copy()
        
        # Save predictions and gradients
        Preds = [x_pred.flatten()]
        f_t_grads = np.zeros(x_pred.shape)
        
        # Obtain data
        # Feedback is revealed only after committing the current decision.
        
        for t in range(T):
            
            # Observe f_t and g_t
            H, y, A, b = episode.step(x_pred)
            f_t = 1/2 * np.linalg.norm(H @ x_pred - y)**2
            g_t = A @ x_pred - b
            g_t_idx = np.argmax(g_t)
            g_t = max(g_t[g_t_idx], 0.0)
            
            # Construct gradient of cost function
            f_grad = (H @ x_pred - y).T @ H
            f_grad = f_grad.reshape(-1,1)
            
            # Gradient step
            x_pred = x_pred - (1/np.sqrt(t+1)) * f_grad
            
            # Projection Step
            data = (x_pred, A[g_t_idx], b[g_t_idx])
            x_pred = self.projection_step(data)
            
            ## Save decision
            Preds.append(x_pred.flatten())
        
        # Return last decision and set of decisions
        return x_pred, Preds
