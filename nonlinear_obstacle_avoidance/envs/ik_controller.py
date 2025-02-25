import numpy as np
import mujoco
import mujoco.viewer as viewer
import mediapy as media

class LevenbegMarquardtIK:
    
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr, damping):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.damping = damping
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    #Levenberg-Marquardt pseudocode implementation
    def calculate(self, goal, init_q, body_id):
        """Calculate the desire joints angles for goal"""
        self.data.qpos = init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)
        delta_q_norm = 0
        while (np.linalg.norm(error) >= self.tol):
            #calculate jacobian
            # return the translation and rotation jacobian for the body to reach the goal
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            #calculate delta of joint q
            n = self.jacp.shape[1]
            I = np.identity(n)
            product = self.jacp.T @ self.jacp + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T
            
            delta_q = j_inv @ error
            #compute next step
            self.data.qpos += self.step_size * delta_q
            #check limits
            self.check_joint_limits(self.data.qpos)
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #calculate new error
            error = np.subtract(goal, self.data.body(body_id).xpos)
            error_norm = np.linalg.norm(error)
            delta_q_norm1 = np.linalg.norm(delta_q)
            # if the 
            # if abs(delta_q_norm1 - delta_q_norm) < 0.001:
            #     break
            delta_q_norm = delta_q_norm1
        return self.data     

    def show(self, model,data, qpos0, target, renderer, camera):
        #Init variables.
        #body_id = model.body('wrist_3_link').id
        body_id = model.body('hand').id
        jacp = np.zeros((3, model.nv)) #translation jacobian
        jacr = np.zeros((3, model.nv)) #rotational jacobian
        goal = [0.49, 0.13, 0.59]
        step_size = 0.5
        tol = 0.01
        alpha = 0.5
        #init_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        init_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        damping = 0.15

        ik = LevenbegMarquardtIK(model, data, step_size, tol, alpha, jacp, jacr, damping)

        #Get desire point
        mujoco.mj_resetDataKeyframe(model, data, 1) #reset qpos to initial value
        ik.calculate(goal, init_q, body_id) #calculate the qpos

        result = data.qpos.copy()

        #Plot results
        print("Results")
        data.qpos = qpos0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        target_plot = renderer.render()

        data.qpos = result
        mujoco.mj_forward(model, data)
        #result_point = data.body('wrist_3_link').xpos
        result_point = data.body('hand').xpos
        renderer.update_scene(data, camera)
        result_plot = renderer.render()

        print("testing point =>", target)
        print("Levenberg-Marquardt result =>", result_point, "\n")

        images = {
            'Testing point': target_plot,
            'Levenberg-Marquardt result': result_plot,
        }

        media.show_images(images)