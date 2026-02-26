import matplotlib.pyplot as plt

def plot_result(test_acc_history, train_acc_history, test_loss_history, train_loss_history):
    fig, ax = plt.subplots(2,2,figsize = (8,6),sharex=True)

    ax[0,0].set_title('Test')
    ax[0,1].set_title('Train')


    ax[0,1].sharey(ax[0,0])
    ax[1,1].sharey(ax[1,0])

    fig.supxlabel("Epoch")



    ax[0,0].set_ylabel('Accuracy')
    ax[1,0].set_ylabel('Loss')

    ax[0,0].plot(test_acc_history,'b.-',label="Fully Connected")
    # ax[0,0].legend()
    ax[0,1].plot(train_acc_history,'b.-')


    ax[1,0].plot(test_loss_history,'b.-')
    ax[1,1].plot(train_loss_history,'b.-')

    plt.tight_layout()
    plt.show()