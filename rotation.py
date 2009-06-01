"""Utilities for fitting ellipses and rotating cell images into alignment."""

from functools import wraps
import numpy as np
import scipy.linalg
import scipy.ndimage as ndimage


class EllipseFitError(ValueError):
    """
    Exception thrown if, for some reason, no ellipse can be fit to the
    given object. 
    """


class EllipseAlignmentError(ValueError):
    """
    Exception thrown when the alignment phase fails; that is, something
    goes wrong once we've fit the ellipse, while aligning.
    """
    pass


def _define_ellipse_constraint(func):
    """
    Wraps so that we only define the constraint matrix once, but avoid
    using global variables.
    """
    constraint = np.zeros((6, 6))
    constraint[2, 0] = -2
    constraint[0, 2] = -2
    constraint[1, 1] =  1
    
    @wraps(func)
    def newfunc(xcoord, ycoord):
        """Dummy docstring."""
        return func(xcoord, ycoord, constraint=constraint)
    
    return newfunc


@_define_ellipse_constraint
def fit_ellipse(xcoord, ycoord, constraint=None):
    """
    Directly fit an ellipse to some scattered 2-D data.
    
    This function fits an ellipse to some scattered 2-D data using the method
    of Fitzgibbon, Pilu and Fisher (1999), minimizing algebraic distance 
    subject to 4ac^2 - b = 1.
    
    Keyword arguments:
    xcoord -- numpy.array of x coordinates (1-dimensional)
    ycoord -- numpy.array of y coordinates (1-dimensional)
    
    Returns:
    Array of coefficients of the ellipse in the order (x^2, xy, y^2, x, y, 1)
    
    This code was directly adapted from MATLAB(tm) code provided in-text by
    the original authors of this method.
    """
    
    
    expansion = np.concatenate(((xcoord**2)[:, np.newaxis], 
        (xcoord * ycoord)[:, np.newaxis], (ycoord * ycoord)[:, np.newaxis], 
        xcoord[:, np.newaxis], ycoord[:, np.newaxis], 
        np.ones((len(ycoord), 1))), axis=1)
    
    scatter = np.dot(expansion.T, expansion)
    
    geval, gevec = scipy.linalg.eig(scatter, constraint)
    negative_c = np.where((geval < 0) & (~np.isinf(geval)))
    return np.real(gevec[:, negative_c]).squeeze()


def radians_to_degrees(angle):
    """Encapsulate turning radians into degrees for code clarity."""
    return angle * 180.0 / np.pi


def align_image_to_ellipse(coeffs, image):
    """
    Given the coefficients of an ellipse in 2D and a binary 
    image, return the angle required to align the image to the
    principal axes of the ellipse (with the longest axis
    as the first major 'hump' on the left).
    """
    
    coeff_a, coeff_b, coeff_c = coeffs[:3]
        
    # Calculate tan(angle) for the angle of rotation of the major axis
    preangle = coeff_b / (coeff_a - coeff_c)
    
    if not np.isinf(preangle):
        # Take the arctan and convert to degrees, which is what 
        # ndimage.rotate uses.
        angle = radians_to_degrees(-0.5 * np.arctan(preangle))
        
        # Order = 0 prevents interpolation from being done and screwing 
        # with our object boundaries.
        rotated = ndimage.rotate(image, angle, order=0)
        
        # Pull out the height/width of just the object.
        try:    
            height, width = rotated[ndimage.find_objects(rotated)[0]].shape
        except IndexError:
            raise EllipseAlignmentError("Can't find object after " \
                + "initial rotation.")
    else:
        angle = 0.
        height, width = image.shape
    
    # we want the height (first axis) to be the major axis.
    if width > height:
        angle -= 90.0
        rotated = ndimage.rotate(image, angle, order=0)
    
    # Correct so that in budding cells, the "major" hump is always
    # on the first.          
    if np.argmax(rotated.sum(axis=1)) > rotated.shape[0] // 2:
        angle -= 180.0
        rotated = ndimage.rotate(image, angle, order=0)
    
    # Do a find_objects on the resultant array after rotation in 
    # order to _just_ get the object and not any of the extra 
    # space that's been added.
    try:
        bounds = ndimage.find_objects(rotated)[0]
    except IndexError:
        raise EllipseAlignmentError("Can't find object after final rotation.")
    
    return rotated[bounds], angle
