import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

__DEBUG__ = False

def show_img(img, cmp = None):
    if not __DEBUG__:
        return
    if cmp:
        plt.imshow(img, cmap = cmp)
    else:
        plt.imshow(img)
    plt.show()

def show_img_2(img, cmp = None):
    if cmp:
        plt.imshow(img, cmap = cmp)
    else:
        plt.imshow(img)
    plt.show()



def draw_res(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset):
    """
    draw the result to the images
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    color_warp = np.zeros((720, 1280, 3), dtype='uint8')

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Annotate lane curvature values and vehicle offset from center
    avg_curve = (left_curve + right_curve)/2
    label_str = 'Curvature Radius: %.1f m' % avg_curve
    result = cv2.putText(result, label_str, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    label_str = 'Vehicle offset: %.1f m' % vehicle_offset
    result = cv2.putText(result, label_str, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    return result
