import attr
from collections import namedtuple
import numpy as np

FrameSize = namedtuple("FrameSize", "w h")

class PinholeCamera:
    def set(self, w, h, f, c, R, t):
        """
        (w,h): width, height in pixels
        f: (fx,fy) focal length in pixels
        c: (cx,cy) principle point in pixels (image center)
        R: rotation from world to camera
        t: translation from world to camera in meters
        """
        self.shape = FrameSize(w,h)
        Rt = np.hstack((R,t))
        fx, fy = f
        cx, cy = c
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        self.proj_mat = K @ Rt

    def set2(self, **params):
        if "f" in params and "c" in params:
            fx, fy = params["f"]
            cx, cy = params["f"]

            K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ])

        if "R" in params and "t" in params:
            R = np.array(params["R"])
            t = np.array(params["t"])

            Rt = np.hstack((R,t))

        if "width" in params and "height" in params:
            self.shape = FrameSize(params["width"], params["height"])

        if "pi" in params:
            if params["pi"] != "v2.1":
                raise Exception("Wrong PiCamera model:", params["pi"])
                
            w,h = self.shape
            fx = 3.11*w/(3936e-3)
            fy = 3.11*h/(2460e-3)
            cx = w/2
            cy = h/2
            self.K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ])

        self.proj_mat = K @ Rt

    def set_piv2(self, w, h, R=None, t=None):
        """
        Assumptions made using Pi Camera v2 specs, YMMV

            f = 3.11 mm
            sensor = 3936x2460 um

        (w,h): width, height in pixels
        R: rotation[3x3] from world to camera
        t: translation[3,1] from world to camera in meters
        """
        self.shape = FrameSize(w,h)

        if t is None:
            t = np.array([0,0,0]).reshape((3,1))
        elif t.shape == (3,):
            t = t.reshape((3,1))

        if R is None:
            R = np.eye(3)

        self.Rt = np.hstack((R,t))
        fx = 3.11*w/(3936e-3)
        fy = 3.11*h/(2460e-3)
        cx = w/2
        cy = h/2
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        self.proj_mat = self.K @ self.Rt

    # def world2camera(self, points):
    #     """Transforms n points from world coordinates (X) to camera
    #     coordinates (P).
    #
    #     P[3xn] = [R|t][3x4]*X[4xn]
    #
    #     points: world points X[3xn] or X[4xn] where each point
    #             is (x,y,z) or (x,y,z,1) respectively
    #     return: camera points P[3xn]
    #     """
    #     if points.shape[0] == 3:
    #         n = points.shape[1]
    #         points = np.hstack((points, np.ones(n).reshape((n,1))))
    #     return self.Rt @ points

    def forward_project(self, points):
        """
        Transforms world coordinates (P) to 2D image coordinates (p)
        using the projection matrix (K*[R|t]).

        p[3xn] = K[3x3]*[R|t][3x4]*P[4xn]
        where:
            P = (x,y,z,1)
            p = (u,v,1)

        points: camera space(x,y,z)[3, n], this will change to [4xn]
                homogenious coordinates
        returns: points in image space(u,v)[2, n]
        """

        # num_pts = points.shape[1]

        # Change to homogenous coordinate
        points = np.vstack((points, np.ones((1, num_pts))))
        points = self.proj_mat @ points
        points /= points[2, :]
        return points[:2,:]

    def back_project(self, points):
        return None
