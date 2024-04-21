import random
import numpy as np

import matplotlib.pyplot as plt
def plotting(result, ax, test, idx, moving=0, color=None, prob=None):
    if (result == 1).sum() == 0:
        return

    start_r = np.where(result == 1)[0][0]
    start_t = np.where(test == 1)[0][0]
    end_r = np.where(result == 1)[0][-1]
    end_t = np.where(test == 1)[0][-1]

    length_r = end_r - start_r + 1
    length_t = end_t - start_t + 1

    if color is None:
        color1 = 'b'
        color2 = 'r'
    else:
        color1 = color
        color2 = color

    plt.plot([start_t, start_t + length_t], [0 + moving, 0 + moving], color1, linewidth=15, solid_capstyle="butt",
             alpha=0.6)

    start_text = start_t + int(length_t / 2)
    plt.text(start_text, 0 + moving, str(idx), weight='bold', size="x-large", horizontalalignment='center',
             verticalalignment='center')

    if prob is None:
        plt.plot([start_r, start_r + length_r], [-1.5 + moving, -1.5 + moving], color2, linewidth=15,
                 solid_capstyle="butt", alpha=0.6)
        start_text = start_r + int(length_r / 2)
        plt.text(start_text, -1.5 + moving, str(idx), weight='bold', size="x-large", horizontalalignment='center',
                 verticalalignment='center')

    # plt.title(str("result" + str(result.sum())) + "test:" +  str(test.sum()) + "sr:" + str(start_r) + "s_t" + str(start_t))


def plotting_prob(result, ax, test, result_prob, color, moving=0):
    if (result == 1).sum() == 0:
        return
    if color is None:
        color1 = 'b'
        color2 = 'r'
    else:
        color1 = color
        color2 = color
    plt.subplot(ax)
    plt.title(str(result.sum()) + "test:" + str(test.sum()))

    start_r = np.where(result == 1)[0][0]
    start_t = np.where(test == 1)[0][0]
    end_r = np.where(result == 1)[0][0]
    end_t = np.where(test == 1)[0][0]

    length_r = end_r - end_r
    length_t = end_t - end_t
    leng = 600

    #     plt.plot([0, leng], [0, 0], 'g', linewidth=3)
    print()
    #     plt.plot([start_t, start_t + length_t], [0 + moving, 0 + moving], color1, linewidth=15,solid_capstyle = "butt",alpha = 0.6)
    plt.scatter(list(np.arange(0, leng, 1, dtype=int)), list(np.full((leng), -1, dtype=int) - 0.5),
                c=(result_prob * 255).astype(int), alpha=0.6)


### kad hoces da radis sa probab, onda zakomentarises iscrtavanje feature na drugoj liniji, odkomentarises, plotting prob, i
# u ploting zakomentarises crtanje druge linije
def plot_results_complete_layout(start, results_array, test_labels_array, moving_step=0, colors=None, features=None,
                                 result_prob=None):
    plt.clf()
    plt.figure(figsize=(10, 10))
    idd = 0
    for i in range(start, start + 6):
        axx = idd + 321
        plt.subplot(axx)
        moving = 0
        for idx in range(0, len(results_array)):
            color = None if colors is None else colors[idx]
            plotting(results_array[idx][i], axx, test_labels_array[idx][i], idx + 1, moving, color, result_prob)

            if result_prob is not None:
                if (idx == 1):
                    plotting_prob(results_array[idx][i], axx, test_labels_array[idx][i], result_prob[idx][i], color, 0)
                if features is not None:
                    feat = np.where(features[i] == 1)[0][0]
                    plt.plot([feat], [0 + moving], 'o', color="k")
            else:
                if features is not None:
                    feat = np.where(features[i] == 1)[0][0]
                    plt.plot([feat], [0 + moving], 'o', color="k")
                    plt.plot([feat], [-1.5 + moving], 'o', color="k")

            plt.subplot(axx).set_xlim(1, 600)
            moving = moving_step + moving

        idd += 1