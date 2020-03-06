List of modifications: 
* Scale 
* Rotation 
* Noise 
* Blur 
* Motion blur 
* Hue (optional), saturation, brightness 
* Perspective transform 
* Shifts 

 
Implementation: 

0. Each function discussed below takes an image (ndarray) as input, and returns a modified image as output. 

1. Wrote functions that deterministically modify images based on input parameters. This was done with cv2, scikit image, and online resources 

2. Wrote functions which call on the deterministic functions in 1, using python's random.gauss(mean, std_dev) method in order to pick input parameters to the deterministic functions. These functions apply the modifications listed above. 

3. Wrote a function which will, given an input image, will apply one randomly selected function from 2 to the image. Subsequently, it has a 50% chance of returning the image, and a 50% chance of applying another modification (and so on after each modification). 

4. Wrote a script that runs the function from 3 a settable number of times for each image in the input directory, and save them to an output directory 

NOTE: online examples were used for point 1 and 2 above. Sources are cited as comments in the code.