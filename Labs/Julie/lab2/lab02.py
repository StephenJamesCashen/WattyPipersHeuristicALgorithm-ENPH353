import line_following_utils as line_utils 
import video_parser_utils as vid_utils
import cv2


vid = vid_utils.VideoParserUtils('raw_video_feed.mp4')
line_follower = line_utils.LineFollowingUtils()
ret, frame = vid.get_next_frame()
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (frame.shape[1], frame.shape[0]))

while ret:
    # thresh = line_follower.threshold_image(frame, 100)
    x, y = line_follower.center_of_gravity(frame)
    cv2.circle(frame, (x, y), 15, (0, 0, 255), thickness=10)
    out.write(frame)
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break
    ret, frame = vid.get_next_frame()

vid.close()
