from Tensorforce.utils import *

trainingTimeDefault_hist = np.load('default_tensorforce_trainingTimesHistEvaluation.npy')
rewardDefault = np.load('default_tensorforce_rewards.npy')
trainingTimeDefault = np.load('default_tensorforce_trainingTimes.npy')

trainingTimeDefaultNoEdge_hist = np.load('defaultNoEdge_tensorforce_trainingTimesHistEvaluation.npy')
rewardDefaultNoEdge = np.load('defaultNoEdge_tensorforce_rewards.npy')
trainingTimeDefaultNoEdge = np.load('defaultNoEdge_tensorforce_trainingTimes.npy')

trainingTimefedAdapt_hist = np.load('fedAdapt_ppo_trainingTimesHistEvaluation.npy')
rewardfedAdapt = np.load('fedAdapt_ppo_rewards.npy')
trainingTimefedAdapt = np.load('fedAdapt_ppo_trainingTimes.npy')

trainingTimeRandom_hist = np.load('default_random_trainingTimes.npy')

trainingTimeFirstFit_hist = np.load('default_firstFit_trainingTimes.npy')
print(len(rewardDefault))
x = [i for i in range(2001)]
xdefault = [i for i in range(501)]
# draw_hist(title='TrainingTime of IoT Devices',
#           x=trainingTimeDefault_hist,
#           xlabel="TrainingTime",
#           savePath='test',
#           pictureName='default_TrainingTime_hist')

# Create a plot
plt.figure(figsize=(int(10), int(5)))  # Set the figure size
plt.plot(xdefault, trainingTimeDefault, color='red', label='Our Method')
plt.plot(x, trainingTimeDefaultNoEdge, color='green', label='Our Method without edge server')
plt.plot(x, trainingTimefedAdapt, color='blue', label='Fed Adapt')
plt.legend()
plt.title("Training Times")
plt.xlabel("Episode")
plt.ylabel("Training time")
plt.savefig(os.path.join('test', f"trainingTime_netVariation_10_50Device"))
plt.close()

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
axs.violinplot([trainingTimeDefault_hist,
                trainingTimeDefaultNoEdge_hist,
                trainingTimefedAdapt_hist,
                trainingTimeRandom_hist,
                trainingTimeFirstFit_hist],
               showmeans=True,
               showmedians=False)
axs.set_title('Training time of trained models')
axs.yaxis.grid(True)
axs.set_xticks([y + 1 for y in range(5)],
               labels=[f'OptiSplit \nwith edge',
                       f'OptiSplit \nwithout edge',
                       'FedAdapt',
                       f'Random Splitting \nwith edge',
                       'First fit'])
axs.set_xlabel('Trained models')
axs.set_ylabel('Training time')
plt.savefig(os.path.join('test', f"violinplot_netVariation10_50Device"))
plt.close()