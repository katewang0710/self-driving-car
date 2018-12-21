import camera_calibration
import gradient_color_filter
from  util import *
import cv2
import poly_lane_fit
from Line import *

from moviepy.editor import VideoFileClip

def draw_points(img, src_pts):
    p = 0
    for src_pt in src_pts:
        x = int(src_pt[0])
        y = int(src_pt[1])

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(img, str(x) + ":" + str(y) + ":" + str(p),
                (x, y),
                font,
                fontScale,
                fontColor,
                lineType)
        p+=1
        cv2.circle(img, (int(x), int(y)), 10, [255, 0, 0], thickness=10, lineType=8,
                   shift=0)

def get_car_offset(img_sz_x, img_sz_y, p_left_fit_cr, p_right_fit_cr):
    y = img_sz_y - 5

    p_left_most_x = p_left_fit_cr[0] * y ** 2 + p_left_fit_cr[1] * y + p_left_fit_cr[2]
    p_right_most_x = p_right_fit_cr[0] * y ** 2 + p_right_fit_cr[1] * y + p_right_fit_cr[2]

    off1 = p_right_most_x - p_left_most_x
    off2 = img_sz_x /2

    return (off1 - off2) * (3.7 / 700)


def measure_curvature_real(p_left_fit_cr, p_right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    p_left_x =  p_left_fit_cr[0] * ploty **2  + p_left_fit_cr[1]* ploty + p_left_fit_cr[2]
    p_right_x = p_right_fit_cr[0] * ploty ** 2 + p_right_fit_cr[1] * ploty + p_right_fit_cr[2]

    left_fit_cr = np.polyfit(ploty * ym_per_pix, p_left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, p_right_x * xm_per_pix, 2)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad

left_line = Line()
right_line = Line()

def process_img(img):
    #img =  mpimg.imread(img_path)

    undistort_img = cv2.undistort(img, camera_calibration.mtx, camera_calibration.dist, None, camera_calibration.mtx)
    show_img(undistort_img)
    threshed_img = gradient_color_filter.gradint_thresh_combo(undistort_img)

    #show_img(undistort_img)

    img_size = (img.shape[1], img.shape[0])

    #print(img_size)

    src_pts = [[202, 721],
         [1101, 720],
         [594, 451],
         [685, 450]]
    src = np.float32(src_pts)

    dst = np.float32(
        [[300, 720],
         [980, 720],
         [300, 0],
         [980, 0]])

    #draw_points(undistort_img, src_pts)
    show_img(threshed_img)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(threshed_img, M, img_size, flags = cv2.INTER_LINEAR)

    M_reverse = cv2.getPerspectiveTransform(dst, src)


    poly_lane_fit.fit_polynomial(warped, left_line, right_line)

    left_line.fit = left_line.get_best_fit()
    right_line.fit = right_line.get_best_fit()

    v_offset = get_car_offset(img.shape[1], img.shape[0], left_line.fit, right_line.fit)
    left_curv, right_curv = measure_curvature_real(left_line.fit, right_line.fit)
    final_res = draw_res(undistort_img, left_line.fit, right_line.fit, M_reverse, \
                          left_curve = left_curv, right_curve = right_curv, \
                          vehicle_offset = v_offset)

    show_img(final_res)
    return final_res

def load_video(input_file):
    #video = VideoFileClip(input_file).subclip(20, 22)
    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(process_img)
    annotated_video.write_videofile("video_after_process.mp4", audio=False)


if __name__ == '__main__':

    camera_calibration.calibrate_camera()
    #load_video('project_video.mp4')
    load_video('project_video.mp4')
    #load_video('challenge_video.mp4')
    img_file = './test_images/test1.jpg'
    test_img = mpimg.imread(img_file)
    demo_img = process_img(test_img)
    show_img_2(demo_img)





#camera_calibration.calibrate_camera()
##get_warped_img("/Users/xingwang/udacity/udacity_self_driving_car_p1/home/CarND-LaneLines-P1/test_images/solidWhiteRight.jpg")
#get_warped_img("/Users/xingwang/test1.jpg")

#binary_warped = get_warped_img("./examples/signs_vehicles_xygrad.png")
#binary_warped = process_img("./examples/color-shadow-example.jpg")
#poly_img = poly_lane_fit.fit_polynomial(binary_warped)
#poly_img = poly_lane_fit.fit_polynomial_convolve(binary_warped)
#show_img(poly_img)
#i = 10
#get_warped_img("./examples/color-shadow-example.jpg")