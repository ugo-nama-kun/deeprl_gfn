import pathlib

import numpy as np

import matplotlib.pyplot as plt
import pandas
import seaborn as sns


sns.set()
sns.set_context("talk")

fig = plt.figure(figsize=(15, 12))
mode_list = ["CD", "ED", "NI"]


def nannorm2(x):
    s = np.nansum(x ** 2)
    return np.sqrt(s)


for index, mode in enumerate(mode_list):
    FILE_PATH = pathlib.Path("data")
    if mode == "CD":
        FILE_PATH = FILE_PATH / "cd" / "trp-homeostatic_shaped2022-09-19-22-12-55default-10-10-2022-11-15-22-26-44"
    elif mode == "ED":
        FILE_PATH = FILE_PATH / "ed" / "trp-homeostatic_shaped2022-10-18-20-59-40exchange-10-10-2022-11-08-23-37-33"
    elif mode == "NI":
        FILE_PATH = FILE_PATH / "ni" / "trp-homeostatic_shaped2022-09-23-00-16-16default-10-10NI-red-2022-11-16-04-13-06"
    print(FILE_PATH)
    
    n_sample = 20
    MAX_STEPS = 10000
    n_blue_red = (10, 10)
    balance_min, balance_max = 0.02, 0.08
    num_balance = 15
    use_extend_settings = True
    
    balance_array = np.linspace(balance_min, balance_max, num_balance)
    
    cum_nutrient = np.load(str(FILE_PATH / f"cum_nutrient50_19.npy"))
    cum_foods = np.load(str(FILE_PATH / f"cum_foods50_19.npy"))
    intero_data = np.load(str(FILE_PATH / f"cum_nutrient50_19.npy"))
    
    is_dead = np.load(str(FILE_PATH / f"is_dead50_19.npy"))
    final_step = np.load(str(FILE_PATH / f"final_step50_19.npy"))
    final_intero = np.load(str(FILE_PATH / f"final_intero50_19.npy"))
    is_deficit_dead = np.load(str(FILE_PATH / f"is_deficit_dead50_19.npy"))
    
    cum_nutrient_target = np.load(str(FILE_PATH / f"cum_nutrient_target50.npy"))
    cum_foods_target = np.load(str(FILE_PATH / f"cum_foods_target50.npy"))
    intero_data_target = np.load(str(FILE_PATH / f"intero_data_cn_target50.npy"))
    is_dead_target = np.load(str(FILE_PATH / f"is_dead_target50.npy"))
    
    
    '''
    remove data when dead
    '''
    
    def progress_plot2(time_at=-1):
        ax = plt.subplot(3, 3, index + 1)
        
        if mode == "CD":
            col = "r"
        elif mode == "NI":
            col = "g"
        elif mode == "ED":
            col = "b"
        
        if time_at < 0:
            alive_cum_nutrient = np.where(is_dead[:, :, np.newaxis, np.newaxis], np.nan, cum_nutrient)
        else:
            is_dead_finalstep = final_step < time_at
            alive_cum_nutrient = np.where(is_dead_finalstep[:, :, np.newaxis, np.newaxis], np.nan, cum_nutrient)

        mean_cn_b = np.nanmean(alive_cum_nutrient[:, :, time_at, 0], axis=0)
        mean_cn_r = np.nanmean(alive_cum_nutrient[:, :, time_at, 1], axis=0)
        
        std_cn = np.zeros(balance_array.size)
        for i in range(balance_array.size):
            tmpb = alive_cum_nutrient[:, i, time_at, 0] - mean_cn_b[i]
            tmpr = alive_cum_nutrient[:, i, time_at, 1] - mean_cn_r[i]
            std_cn[i] = np.sqrt((nannorm2(tmpb) + nannorm2(tmpr)) / n_sample)
        
        alive_cum_nutrient_target = np.where(is_dead_target[:, np.newaxis, np.newaxis], np.nan, cum_nutrient_target)
        mean_cn_tb = np.nanmean(alive_cum_nutrient_target[:, time_at, 0])
        mean_cn_tr = np.nanmean(alive_cum_nutrient_target[:, time_at, 1])
        std_cn_tb = np.nanstd(alive_cum_nutrient_target[:, time_at, 0])
        std_cn_tr = np.nanstd(alive_cum_nutrient_target[:, time_at, 1])
        
        # plot error bar
        for i in range(balance_array.size):
            # nurtient rail
            coeff = 50 * np.max(cum_nutrient)
            plt.plot([0, coeff * balance_array[i]], [0, coeff * (0.1 - balance_array[i])], "--", color=(0, 0, 0, 0.1), linewidth=1)

            err_angle = np.arctan2(0.1 - balance_array[i], balance_array[i])
            dx = std_cn[i] * np.cos(err_angle)
            dy = std_cn[i] * np.sin(err_angle)
            plt.plot([mean_cn_b[i], mean_cn_b[i] + dx], [mean_cn_r[i], mean_cn_r[i] + dy], col, linewidth=2, alpha=0.8)
            plt.plot([mean_cn_b[i], mean_cn_b[i] - dx], [mean_cn_r[i], mean_cn_r[i] - dy], col, linewidth=2, alpha=0.8)
        
        dx = std_cn_tb
        dy = std_cn_tr

        # plot mean points
        plt.scatter(mean_cn_b, mean_cn_r, c=col, s=20)

        # plot target
        plt.plot([mean_cn_tb, mean_cn_tb + dx], [mean_cn_tr, mean_cn_tr], "k", alpha=1)
        plt.plot([mean_cn_tb, mean_cn_tb - dx], [mean_cn_tr, mean_cn_tr], "k", alpha=1)
        plt.plot([mean_cn_tb, mean_cn_tb], [mean_cn_tr, mean_cn_tr + dy], "k", alpha=1)
        plt.plot([mean_cn_tb, mean_cn_tb], [mean_cn_tr, mean_cn_tr - dy], "k", alpha=1)
        plt.scatter([mean_cn_tb], [mean_cn_tr], c="k", s=20, alpha=1)
        
        # Ideal
        if mode == "CD":
            def cd_ideal():
                c_x, c_y = mean_cn_tb / 2, mean_cn_tr / 2
                theta = np.linspace(0, 2*np.pi, 100)
                r = np.linalg.norm([c_y, c_x])
                x_ = c_x + r * np.cos(theta)
                y_ = c_y + r * np.sin(theta)
                return x_, y_
            
            x_, y_ = cd_ideal()
            plt.plot(x_, y_, "k", alpha=0.2)
            
            plt.xlabel("Accumulated\n Blue")
            plt.ylabel("Accumulated\n Red")
        
        elif mode == "ED":
            def ed_ideal_at(x):
                return -1 * (x - mean_cn_tb) + mean_cn_tr

            plt.plot([0, 3.0], [ed_ideal_at(0), ed_ideal_at(3)], "k", alpha=0.2)

        elif mode == "NI":
            plt.plot([mean_cn_tb, mean_cn_tb], [0, 5], "k", alpha=0.2)
        
        # plt.legend(balance_list)
        ax.set_aspect('equal', 'box')

        max_lim = np.nanmax(alive_cum_nutrient[:, :, time_at, :]) + 0.1
        if col != "g":
            plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0])
            plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
            plt.ylim([0, 2.1])
            plt.xlim([0, 2.1])
        else:
            plt.ylim([0, max_lim])
            plt.xlim([0, max_lim])
        
        print("all, min:{}, max:{}".format(np.min(cum_nutrient), np.max(cum_nutrient[:, :, time_at, :])))
        print("alive, min:{}, max:{}".format(np.nanmin(alive_cum_nutrient), np.nanmax(alive_cum_nutrient[:, :, time_at, :])))
        
        # error plots
        ax = plt.subplot(3, 3, index + 1 + 3)
        
        x = np.arctan2(balance_array, 0.1 - balance_array)
        err_b = 100 * (mean_cn_b[::-1] - mean_cn_tb) / mean_cn_tb
        err_r = 100 * (mean_cn_r[::-1] - mean_cn_tr) / mean_cn_tr

        plt.plot(x, err_b, "b")
        plt.scatter(x, err_b, c=["b"] * len(x))
        plt.plot(x, err_r, "r")
        plt.scatter(x, err_r, c=["r"] * len(x))
        
        # Target
        plt.plot([np.arctan2(mean_cn_tr, mean_cn_tb)], [0.0], c="k", marker="x", markersize=15, markeredgewidth=3)
        
        if mode == "CD":
            def cd_ideal():
                theta = np.linspace(-np.pi / 4, np.pi * 3 / 4, 100)
                c_x, c_y = mean_cn_tb / 2, mean_cn_tr / 2
                r = np.linalg.norm([c_x, c_y])
                v_b = c_x + r * np.cos(theta)
                v_r = c_y + r * np.sin(theta)
                a_ = np.arctan2(v_r, v_b)
                return v_b, v_r, a_
            
            v_b_, v_r_, a_ = cd_ideal()
            err_b = 100 * (v_b_ - mean_cn_tb) / mean_cn_tb
            err_r = 100 * (v_r_ - mean_cn_tr) / mean_cn_tr
            plt.plot(a_, err_r, "r", alpha=0.2)
            plt.plot(a_, err_b, "b", alpha=0.2)
            
            plt.xlabel("Rail Angle [rad]")
            plt.ylabel("Error")
        
        elif mode == "ED":
            def ed_ideal():
                v_b = np.linspace(0.0, 3.0, 100)
                v_r = -1 * (v_b - mean_cn_tb) + mean_cn_tr
                a_ = np.arctan2(v_r, v_b)
                return v_b, v_r, a_

            v_b_, v_r_, a_ = ed_ideal()
            err_b = 100 * (v_b_ - mean_cn_tb) / mean_cn_tb
            err_r = 100 * (v_r_ - mean_cn_tr) / mean_cn_tr
            plt.plot(a_, err_r, "r", alpha=0.2)
            plt.plot(a_, err_b, "b", alpha=0.2)
            
        elif mode == "NI":
            def ni_ideal():
                v_b = np.array([mean_cn_tb] * 100)
                v_r = np.linspace(0, 10, 100)
                a_ = np.arctan2(v_r, v_b)
                return v_b, v_r, a_

            v_b_, v_r_, a_ = ni_ideal()
            err_b = 100 * (v_b_ - mean_cn_tb) / mean_cn_tb
            err_r = 100 * (v_r_ - mean_cn_tr) / mean_cn_tr
            plt.plot(a_, err_r, "r", alpha=0.2)
            plt.plot(a_, err_b, "b", alpha=0.2)

        plt.ylim([-100, 100])
        angle_min, angle_max = np.arctan2(balance_min, 0.1 - balance_min), np.arctan2(balance_max, 0.1 - balance_max)
        plt.xlim([angle_min - 0.15, angle_max + 0.15])
        if mode != "CD":
            plt.yticks([-100, -50, 0, 50, 100], [""] * 5)
        print(angle_min, angle_max)
        ax.set_box_aspect(1)
    
    progress_plot2()
    
    df = pandas.read_csv("raw_data/food_count.csv")
    
    for index, mode in enumerate(["cd", "ed", "ni"]):
        ax = plt.subplot(3, 3, index + 1 + 6)
        ax.set_box_aspect(1)

        ax = sns.violinplot(df[df["mode"] == mode],
                            x="region", y="counts", hue="color", cut=1, linewidth=1.5,
                            palette={"red": "r", "blue": "b"})
        ax.get_legend().remove()
        plt.ylim([-10, 110])
        plt.xlabel("")
        
        if mode != "cd":
            plt.ylabel("")
            # plt.yticks([20, 40, 60, 80, 100], [""] * 5)
        else:
            plt.ylabel("Count of\n Consumed Food")

fig.tight_layout()
plt.savefig("Fig2.pdf")
plt.show()

print("Finish.")
