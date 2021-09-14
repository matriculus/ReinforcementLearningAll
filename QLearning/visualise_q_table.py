import numpy as np
import matplotlib.pyplot as plt
import os, cv2, sys

def make_folder(fname):
    if not os.path.exists(fname):
        os.mkdir(fname)

make_folder("qtable_charts")

def get_sorted_files_with_eps(folder):
    files = os.listdir(folder)
    files = [file for file in files if ".npy" in file]
    episodes = [int(file.split('-')[0]) for file in files]
    numbering = np.argsort(episodes)
    files = [os.path.join(folder, files[num]) for num in numbering]
    return (files, np.sort(episodes))

q_table_folder = "qtables"
files, episodes = get_sorted_files_with_eps(q_table_folder)

# make video
if sys.platform == 'win32':
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
elif 'linux' in sys.platform:
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

make_folder("Video")
out = cv2.VideoWriter(os.path.join("Video",'qlearn.avi'), fourcc, 60.0, (1200, 900))


def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3

fig = plt.figure(figsize=(12, 9))


for file, episode in zip(files, episodes):
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    q_table = np.load(file)

    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
            ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
            ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])

            ax1.set_ylabel("Action 0")
            ax2.set_ylabel("Action 1")
            ax3.set_ylabel("Action 2")

    #plt.show()
    img_path = f"qtable_charts/{episode}.png"
    plt.savefig(img_path)
    frame = cv2.imread(img_path)
    out.write(frame)
    plt.clf()

out.release()