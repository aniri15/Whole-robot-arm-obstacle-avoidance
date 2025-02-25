import ctypes

class PandaIK:
    def __init__(self,
                 path_to_lib='/usr/local/diana/lib/libPanda-IK.so'):
        self.pandaik_lib = ctypes.CDLL(path_to_lib)
        self.pandaik_lib.compute_inverse_kinematics_void.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.pandaik_lib.compute_inverse_kinematics_void.restype = None

    def compute_inverse_kinematics(self, xyzrpy, q_actual):
        output = (ctypes.c_double * 7)()
        q_actual_7 = q_actual[:7]
        xyzrpy_ctypes = xyzrpy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        q_actual_ctypes = q_actual_7.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.pandaik_lib.compute_inverse_kinematics_void(xyzrpy_ctypes, q_actual_ctypes, output)
        # compare the current joint angles with the output,
        # if they are the same, then the solution is not found
        if all(q_actual_7[i] == output[i] for i in range(7)):
            print("Cannot find the solution")
            breakpoint()
        # concentrate the output and the last two elements of q_actual
        output = list(output)
        output.extend(q_actual[7:])
        return output
    
if __name__ == '__main__':
    pandaik = PandaIK()
    xyzrpy = (ctypes.c_double * 6)(0.4, 0.0, 0.6, 45.0, -45.0, 0.0)
    q_actual = (ctypes.c_double * 7)(-0.29, -0.176, -0.232, -0.67, 1.04, 2.56, 0.0)
    output = pandaik.compute_inverse_kinematics(xyzrpy,q_actual)
    print("Result from C++:", output)  # Assuming the function returns an array of 7 doubles
