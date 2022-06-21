from matplotlib.collections import LineCollection
import pint
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#
NUM_ECC_ITER = 20
POLY_DEG_FIT = 5
#
class IncorrectUnit(Exception):
    def __init__(self, inputs, message='Input for the following function was found to be incorrect:'):
        self.inputs = inputs
        inputs_str = ' '
        for input_var in self.inputs:
            if (self.inputs.index(input_var) + 1) != len(self.inputs):
                inputs_str = inputs_str + str(input_var) + ', '
            else:
                inputs_str = inputs_str + str(input_var)
        super().__init__(message + inputs_str)

#
class BearingSolution():
    def __init__(self, diametric_clearance, groove_width, length, number_of_grooves, shaft_diameter, \
        shaft_speed, viscosity):

        # 
        self._diametric_clearance = self._input_check(diametric_clearance_check=diametric_clearance)
        self.groove_width = groove_width
        self.length = length
        self.number_of_grooves = number_of_grooves
        self.shaft_diameter = shaft_diameter
        self.shaft_speed = shaft_speed
        self.viscosity = viscosity
        
        self.iterations = self.get_iterations()
            
    #
    def get_iterations(self):
        solutions = []
        eccentricity_lower_limit = (self.diametric_clearance / (2 * NUM_ECC_ITER))
        eccentricity_upper_limit = (self.diametric_clearance / 2) 
        eccentricity_step = (eccentricity_upper_limit.to('mm').magnitude - eccentricity_lower_limit.to('mm').magnitude) / (NUM_ECC_ITER - 1)
        for i in np.arange(eccentricity_lower_limit.to('mm').magnitude, eccentricity_upper_limit.to('mm').magnitude, eccentricity_step):
            eccentricity = i * ureg.mm
            iterations = []
            for j in np.arange(0.01, 0.99, 0.01):
                conv_div_ratio = j 
                new_iteration = BearingIteration(conv_div_ratio, self.diametric_clearance, eccentricity, \
                    self.groove_width, self.length, self.number_of_grooves, self.shaft_diameter, \
                    self.shaft_speed, self.viscosity)
                iterations.append(new_iteration)
            solutions.append((eccentricity,iterations))
        return solutions

    # diametric clearance getter function
    @property
    def diametric_clearance(self):
        return self._diametric_clearance

    # eccentricity getter function
    @property
    def eccentricity(self):
        return self._eccentricity

    # groove width getter function
    @property
    def groove_width(self):
        return self._groove_width

    # eccentricity setter function
    @eccentricity.setter
    def eccentricity(self, new_eccentricity):
        inputs_check = []
        new_eccentricity.to_base_units()
        if new_eccentricity.dimensionality != '[length]':
            inputs_check.append('BearingLoadCapacityCalculator() <- eccentricity')
            raise IncorrectUnit(inputs_check)
        else:
            self._eccentricity = new_eccentricity

    # diametric clearance setter function
    @diametric_clearance.setter
    def diametric_clearance(self, new_diametric_clearance):
        inputs_check = []
        new_diametric_clearance.to_base_units()
        if new_diametric_clearance.dimensionality != '[length]':
            inputs_check.append('BearingLoadCapacityCalculator() <- diametric_clearance')
            raise IncorrectUnit(inputs_check)
        else:
            self._diametric_clearance = new_diametric_clearance

    # groove width setter function
    @groove_width.setter
    def groove_width(self, new_groove_width):
        inputs_check = []
        new_groove_width.to_base_units()
        if new_groove_width.dimensionality != '[length]':
            inputs_check.append('BearingLoadCapacityCalculator() <- groove_width')
            raise IncorrectUnit(inputs_check)
        else:
            self._groove_width = new_groove_width

    #  
    def _input_check(self, **kwargs):
        input_checks = []
        for key, value in kwargs.items():
            value.to_base_units()
            if key == 'diametric_clearance_check' and value.dimensionality != '[length]':
                input_checks.append('CapacityFilmCalculator() -> diametric_clearance')
        if len(input_checks) > 0:
            raise IncorrectUnit(input_checks)
        if len(kwargs) > 1:
            checked_inputs = kwargs
        else:
            key = [*kwargs][0]
            checked_inputs = kwargs[key]
        return checked_inputs

    # 
    def get_ecc_load_curve(self):
        # 
        eccentricity_points = []
        min_load_points = []
        for iteration in self.iterations:
            eccentricity = iteration[0]
            conv_div_ratio = []
            load_capacity = []
            for ratio_iteration in iteration[1]:
                conv_div_ratio.append(ratio_iteration.conv_div_ratio)
                load_capacity.append(ratio_iteration.load_capacity.to('N').magnitude)
            coeff = np.polynomial.polynomial.polyfit(conv_div_ratio, load_capacity, POLY_DEG_FIT)
            
            poly = np.polynomial.Polynomial(coeff)
            # plt.plot(conv_div_ratio, load_capacity)
            # plt.plot(conv_div_ratio, [poly(ratio) for ratio in conv_div_ratio])
            # plt.show()
            # derivative of solution curve at constant eccentricity
            poly_deriv = poly.deriv()
            # roots of the solution derivative
            poly_deriv_roots = poly_deriv.roots()
            min_root = None
            for root in poly_deriv_roots:
                if min_root == None and poly(root) >= 0:
                    min_root = root
                elif poly(root) >=0 and poly(root) < poly(min_root):
                    min_root = root
            min_load = poly(min_root) * ureg.N
            eccentricity_points.append(eccentricity)
            min_load_points.append(min_load)
        return [eccentricity_points, min_load_points]            

#
class BearingIteration():
    def __init__(self, conv_div_ratio, diametric_clearance, eccentricity, groove_width, length, 
            number_of_grooves, shaft_diameter, shaft_speed, viscosity):
        # 
        self.conv_div_ratio = conv_div_ratio
        self.radial_clearance = self.radial_clearance(diametric_clearance)
        self.eccentricity_ratio = self.eccentricity_ratio(eccentricity, self.radial_clearance)
        self.groove_angle = self.groove_angle(groove_width, shaft_diameter, diametric_clearance)
        self.inner_radius = self.inner_radius(shaft_diameter, self.radial_clearance)
        self.pad_angle = self.pad_angle(number_of_grooves, self.groove_angle)
        self.peripheral_velocity = self.peripheral_velocity(shaft_diameter, shaft_speed)
        self.pads = self.construct_pads(conv_div_ratio, number_of_grooves, viscosity, length)  
        self.load_capacity = self.bearing_load_capacity()
  
    # 
    def construct_pads(self, conv_div_ratio, number_of_grooves, viscosity, length):
        pads = []
        upper_limit = int(number_of_grooves / 2) + 1
        for number in range(1, upper_limit, 1):
            if number == 1:
                trailing_angle = (math.pi * ureg.radians)  
                leading_angle = (math.pi * ureg.radians) - (self.pad_angle * conv_div_ratio)
            else:
                trailing_angle = pads[-1].leading_angle - self.groove_angle
                leading_angle = trailing_angle - self.pad_angle
            arc_length = self.arc_length(self.inner_radius, leading_angle, trailing_angle)
            leading_film_thickness = self.leading_film_thickness(self.eccentricity_ratio, leading_angle, \
                self.radial_clearance)
            trailing_film_thickness = self.trailing_film_thickness(self.eccentricity_ratio, \
                self.radial_clearance, trailing_angle)
            film_thickness_ratio = self.film_thickness_ratio(leading_film_thickness, \
                trailing_film_thickness)
            load_center = self.load_center(film_thickness_ratio, arc_length)
            load_center_ratio = self.load_center_ratio(load_center, arc_length)
            load_capacity = self.pad_load_capacity(arc_length, length, film_thickness_ratio, \
                trailing_film_thickness, self.peripheral_velocity, viscosity)
            load_capacity_x = self.pad_load_capacity_x(leading_angle, load_capacity, load_center_ratio, \
                trailing_angle)
            load_capacity_y = self.pad_load_capacity_y(leading_angle, load_capacity, load_center_ratio, \
                trailing_angle)
            pads.append(BearingPad(arc_length, film_thickness_ratio, leading_angle, \
                leading_film_thickness, load_capacity, load_capacity_x, load_capacity_y, load_center, \
                load_center_ratio, number, trailing_angle, trailing_film_thickness))
        return pads


    # provides the resultant bearing load capacity
    def bearing_load_capacity(self):
        x_sum = 0
        y_sum = 0
        for pad in self.pads:
            x_sum += pad.load_capacity_x
            y_sum += pad.load_capacity_y
        resultant = (x_sum**2 + y_sum**2)**(1/2)
        return resultant

    # method for calculating the arc length of a bearing pad
    def arc_length(self, inner_radius, leading_angle, trailing_angle):
        length = ((trailing_angle - leading_angle) / (2 * math.pi * ureg.radians)) * \
            (math.pi * inner_radius * 2)
        return length

    # calculates and returns the eccentricity ratio
    def eccentricity_ratio(self, eccentricity, radial_clearance):
        ratio = eccentricity / radial_clearance
        return ratio

    # calculates and returns pad leading edge film thickness ratio
    def film_thickness_ratio(self, leading_film_thickness, trailing_film_thickness):
        ratio = leading_film_thickness / trailing_film_thickness
        return ratio

    # calculates and returns the bearing groove angle
    def groove_angle(self, groove_width, shaft_diameter, diametric_clearance):
        angle = (2 * math.atan(groove_width / (shaft_diameter + diametric_clearance))) * \
            ureg.radians
        return angle

    # calculates and returns the bearing inner radius
    def inner_radius(self, shaft_diameter, radial_clearance):
        radius = (shaft_diameter / 2) + radial_clearance
        return radius

    # calculates the leading edge film thickness of a bearing pad
    def leading_film_thickness(self, eccentricity_ratio, leading_angle, radial_clearance):
        film_thickness = radial_clearance * (1 + (eccentricity_ratio * math.cos(leading_angle)))
        return film_thickness

    # finds the load center for a given pad
    def load_center(self, film_ratio, pad_arc_length):
        a = film_ratio * ((film_ratio + 2) / (film_ratio - 1)) * math.log(film_ratio)
        b = (5 / 2) * (film_ratio - 1)
        c = (film_ratio + 1) * math.log(film_ratio)
        d = 2 * (film_ratio - 1)
        load_center = ((a - b - 3) / (c - d)) * pad_arc_length
        return load_center

    # returns the load center ratio for a bearing pad
    def load_center_ratio(self, load_center, arc_length):
        center_ratio = load_center / arc_length
        return center_ratio

    # calculates the load capacity of the given pad
    def pad_load_capacity(self, arc_length, length, film_thickness_ratio, trailing_film_thickness, \
        velocity, viscosity): 
        a = (film_thickness_ratio + 1) * math.log(film_thickness_ratio) - 2 * (film_thickness_ratio - 1)
        b = viscosity * velocity * (arc_length**2) * length 
        c = ((film_thickness_ratio - 1)**2) * (film_thickness_ratio + 1)
        load = ((6 * a) / c) * (b / (trailing_film_thickness**2))
        return load

    # calculates the x component of the pad load capacity
    def pad_load_capacity_x(self, leading_angle, load_capacity, load_center_ratio, trailing_angle):
        load_capacity_x = load_capacity * math.sin(math.pi - (load_center_ratio * trailing_angle + \
            (1 - load_center_ratio) * leading_angle))
        return load_capacity_x

    # calculates the y component of the pad load capacity
    def pad_load_capacity_y(self, leading_angle, load_capacity, load_center_ratio, trailing_angle):
        load_capacity_y = load_capacity * math.cos(math.pi - (load_center_ratio * trailing_angle + \
            (1 - load_center_ratio) * leading_angle))
        return load_capacity_y

    # calculates pad angles
    def pad_angle(self, number_of_grooves, groove_angle):
        angle = (((360 * ureg.degree) / number_of_grooves) - groove_angle)
        return angle

    # calculates the shaft surface speed
    def peripheral_velocity(self, shaft_diameter, shaft_speed):
        velocity = (shaft_diameter / 2) * shaft_speed
        return velocity

    # returns the bearing running clearance
    def radial_clearance(self, diametric_clearance):
        clearance = diametric_clearance / 2
        return clearance

    # calculates pad trailing film thickness
    def trailing_film_thickness(self, eccentricity_ratio, radial_clearance, trailing_angle):
        film_thickness = radial_clearance * (1 + (eccentricity_ratio * math.cos(trailing_angle)))
        return film_thickness
    
#
class BearingPad():
    def __init__(self, arc_length, film_thickness_ratio, leading_angle, leading_film_thickness, \
            load_capacity, load_capacity_x, load_capacity_y, load_center, load_center_ratio, number, \
            trailing_angle, trailing_film_thickness):
        self.arc_length = arc_length
        self.film_thickness_ratio = film_thickness_ratio
        self.leading_angle = leading_angle
        self.leading_film_thickness = leading_film_thickness
        self.load_capacity = load_capacity
        self.load_capacity_x = load_capacity_x
        self.load_capacity_y = load_capacity_y
        self.load_center = load_center
        self.load_center_ratio = load_center_ratio
        self.number = number
        self.trailing_angle = trailing_angle
        self.trailing_film_thickness = trailing_film_thickness

# exponential function with transformation
def expFunc(x, a, b, c):
    return a * np.exp(b * x) + c




ureg = pint.UnitRegistry()

def main():
    
    shaft_diameter = 600 * ureg.mm
    number_of_grooves = 30
    shaft_speed = 200 * ureg.rpm
    diametric_clearance = 0.2 * ureg.mm
    groove_width = 10 * ureg.mm
    bearing_length = 600 * ureg.mm
    viscosity = 0.009967 * ureg.poise

    calc = BearingSolution(diametric_clearance, groove_width, bearing_length, number_of_grooves, \
        shaft_diameter, shaft_speed, viscosity)

    points = calc.get_ecc_load_curve()

    x_val = [ecc_val.to('mm').magnitude for ecc_val in points[0]]
    y_val = [load_val.to('lbf').magnitude for load_val in points[1]]

    popt, pcov = curve_fit(expFunc, x_val, y_val)
    print(type(popt))

    coeff = np.polynomial.polynomial.polyfit(x_val, y_val, 1)
    poly = np.polynomial.Polynomial(coeff)

    plt.plot(x_val, expFunc(x_val, popt[0], popt[1], popt[2]))
    plt.plot(x_val, y_val, 'o')
    plt.plot(x_val, [poly(val) for val in x_val])
    plt.show()

if __name__ == '__main__':
    main()




