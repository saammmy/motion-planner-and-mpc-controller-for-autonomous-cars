from Controller import MPC
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt

save_dict = True
if save_dict:
    dict_poly = {}
    for itr in range(25,101,5):
        throttle = itr/100
        dict_poly[throttle] = np.load("./data/poly_acc_d3/"+str(throttle)+".npy") 
    np.save("./data/poly3_dict", dict_poly)

else:
    throttle = 0.2
    path = "./data/"+ str(throttle)

    speed_list_acc = np.loadtxt(path + "/speed_list_acc.csv", delimiter=",", dtype=float)
    acc_list = np.loadtxt(path + "/acc_list.csv", delimiter=",", dtype=float)

    speed_list_dec = np.loadtxt(path + "/speed_list_dec.csv", delimiter=",", dtype=float)
    dec_list = np.loadtxt(path + "/dec_list.csv", delimiter=",", dtype=float)

    def func_exp(x, a, b, c):
        return a * np.exp(-b * x) + c

    poly_acc_d3 = np.polyfit(speed_list_acc, acc_list, 3)
    poly_acc_d4 = np.polyfit(speed_list_acc, acc_list, 4)

    # coeff_acc_exp, cvar_acc = curve_fit(func_exp, speed_list_acc, acc_list)

    poly_dec_d3 = np.polyfit(speed_list_dec, dec_list, 3)
    poly_dec_d4 = np.polyfit(speed_list_dec, dec_list, 4)

    np.save("./data/poly_acc_d3/"+str(throttle), poly_acc_d3)
    np.save("./data/poly_acc_d4/"+str(throttle), poly_acc_d4)
    # np.save("./data/coeff_acc_exp/"+str(throttle), coeff_acc_exp)

    np.save("./data/poly_dec_d3/"+str(throttle), poly_dec_d3)
    np.save("./data/poly_dec_d4/"+str(throttle), poly_dec_d4)

    print("Throttle = ", throttle)
    print("At 0 Velocity D=3: ", np.polyval(poly_acc_d3,0))
    print("At 30 Velocity D=3: ", np.polyval(poly_acc_d3,30))
    print("At 38 Velocity D=3: ", np.polyval(poly_acc_d3,38))

    # print("At 0 Velocity D=4: ", np.polyval(poly_acc_d4,0))
    # print("At 38 Velocity D=4: ", np.polyval(poly_acc_d4,38))

    # print("At 0 Velocity exp: ",func_exp(0, *coeff_acc_exp))
    # print("At 38 Velocity exp: ", func_exp(38, *coeff_acc_exp))

    plt.figure()
    plt.scatter(speed_list_acc, acc_list)
    plt.plot(speed_list_acc, np.polyval(poly_acc_d3, speed_list_acc), 'r--', label="fit Degree = 3")
    plt.plot(speed_list_acc, np.polyval(poly_acc_d4, speed_list_acc), 'b-.', label="fit Degree = 4")
    # plt.plot(speed_list_acc, func_exp(speed_list_acc, *coeff_acc_exp), 'c:', label="fit Exponential")

    plt.xlabel('Velocity(m/s)')
    plt.ylabel('Acceleration(m/s2)')
    plt.title("Curve Fitting for Acceleration at Throttle = " + str(throttle))
    plt.legend()

    plt.figure()
    plt.scatter(speed_list_dec, dec_list)
    plt.plot(speed_list_dec, np.polyval(poly_dec_d3, speed_list_dec), 'r--', label="fit Degree = 3")
    plt.plot(speed_list_dec, np.polyval(poly_dec_d4, speed_list_dec), 'b-.', label="fit Degree = 4")
    plt.xlabel('Velocity(m/s)')
    plt.ylabel('Acceleration(m/s2)')
    plt.title("Curve Fitting for Deceleration")
    plt.legend()

    plt.show()