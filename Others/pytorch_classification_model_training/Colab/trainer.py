from copy import deepcopy
import numpy as np
from tqdm.auto import tqdm
#import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

class Trainer():
    def __init__(self, config):
        self.save_path = config.save_path
        self.n_epochs = config.n_epochs
        self.batch_size = config.batch_size
        self.class_name = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def _training(self, model, loader, criterion, optimizer, device):
        model.train()
        train_loss = 0.
        train_acc = 0.

        with tqdm(loader, unit = "batch") as tepoch:
            for i, (X, y) in enumerate(tepoch):
                tepoch.set_description('Training')
                X = X.to(device)
                y = y.to(device)

                y_hat = model(X)
                loss = criterion(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss
                iter_loss = loss.item()
                train_loss += iter_loss

                # acc
                pred = y_hat.max(1, keepdim = True)[1]
                iter_acc = pred.eq(y.data.view_as(pred)).sum().item()
                train_acc += iter_acc
                tepoch.set_postfix(train_iter_loss = f'{iter_loss / X.size(0):.3f}',
                                   train_iter_accuracy = f'{iter_acc/ X.size(0)*100:.2f}%')
                
        train_loss = train_loss / len(loader)
        train_acc = train_acc / len(loader.dataset)
        return train_loss, train_acc

    def _evaluation(self, model, loader, criterion, device):
        model.eval()
        val_loss = 0.
        val_acc = 0.
        
        with torch.no_grad():
            with tqdm(loader, unit = 'batch') as tepoch:
                for (X, y) in tepoch:
                    tepoch.set_description('Evaluation')
                    X = X.to(device)
                    y = y.to(device)

                    y_hat = model(X)
                    loss = criterion(y_hat, y)
                    
                    # loss
                    iter_loss = loss.item()
                    val_loss += iter_loss

                    # pred
                    pred = y_hat.max(1, keepdim = True)[1]
                    iter_acc = pred.eq(y.data.view_as(pred)).sum().item()
                    val_acc += iter_acc
                    tepoch.set_postfix(eval_iter_loss = f'{iter_loss / X.size(0):.3f}',
                                       eval_iter_accuracy = f'{iter_acc / X.size(0)*100:.2f}%')
        
        val_loss = val_loss / len(loader)
        val_acc = val_acc / len(loader.dataset)
        return val_loss, val_acc

    def _plotting_training(self, train_loss_list, train_acc_list, val_loss_list, val_acc_list, best_epoch):
        def plotting(train_list, val_list, types, best_epoch):
            plt.plot(train_list, label = f'Train_{types}')
            plt.plot(val_list, label = f'Valid_{types}')
            plt.legend()
            plt.title(f'{types}')
            plt.xlabel('Epochs')
            plt.ylabel(f'{types}', rotation = 360, labelpad = 30)
            plt.axvline(best_epoch, color = 'red')
            plt.xticks([i for i in range(len(train_list))], [i+1 for i in range(len(train_list))])
            plt.grid(True)
        plt.figure(figsize = (20,5))
        plt.subplot(1,2,1)
        plotting(train_loss_list, val_loss_list, 'Loss', best_epoch)
        plt.subplot(1,2,2)
        plotting(train_acc_list, val_acc_list, 'Accuracy', best_epoch)
        plt.savefig('Training_Graph.png')
    
    def _confusion_matrix(self, model, valid_loader, test_loader, device):
        class_name = self.class_name
        def create_confusion_matrix(model, loader, class_name, device):
            labels = torch.Tensor()
            preds = torch.Tensor()

            model.eval()
            with torch.no_grad():
                for img, label in loader:
                    labels = torch.cat([labels, label])
                    img = img.to(device)
                    pred = model(img).max(1, keepdim = True)[1].detach().to('cpu').reshape(-1)
                    preds = torch.cat([preds, pred])

            preds = preds.numpy().reshape(-1)
            labels = labels.numpy().reshape(-1)
            
            conf_df =  pd.DataFrame(confusion_matrix(labels, preds))
            conf_df.columns = class_name
            conf_df.index = class_name
            return conf_df
        
        def create_sns_conf(conf_df, phase):
            plt.figure(figsize = (10,7))
            sns.heatmap(conf_df, annot = True, fmt = 'd', cmap='YlGnBu')
            plt.yticks(rotation = 360)
            plt.xlabel('model pred')
            plt.ylabel('label', rotation = 360, labelpad = 20)
            plt.title(f'confusion matrix ({phase})')
            plt.savefig(f'confusion_matrix_{phase}.png')

        val_conf_df = create_confusion_matrix(model, valid_loader, class_name, device)
        test_conf_df = create_confusion_matrix(model, test_loader, class_name, device)
        
        create_sns_conf(val_conf_df, 'Valid')
        create_sns_conf(test_conf_df, 'Test')
    
    def train(self, model, criterion, optimizer, train_loader, valid_loader, test_loader, device):
        low_loss = np.inf
        high_acc = 0
        best_model = None
        best_epoch = None
        # 시각화용
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        for epoch in range(self.n_epochs):
            print(f'------------------ EPOCH : {epoch + 1}/{self.n_epochs} ------------------')
            train_loss, train_acc = self._training(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = self._evaluation(model, valid_loader, criterion, device)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            print(f"Train_loss={train_loss:.3f} Valid_loss={val_loss:.3f} Train_accuracy={train_acc*100:.2f}% Valid_accuracy={val_acc*100:.2f}%")
            if val_loss < low_loss:
                print(f"* Valid_loss가 향상되었습니다. {low_loss:.3f} ==> {val_loss:.3f}")
                low_loss = val_loss
                high_acc = val_acc
                best_epoch =  epoch
                best_model = deepcopy(model.state_dict())
            else:
                print('* Valid_loss가 향상되지 않았습니다.')
        
        print(f'------------------ 학습종료 ------------------')
        # save training graph
        print('학습 그래프 생성 중....')
        self._plotting_training(train_loss_list, train_acc_list, val_loss_list, val_acc_list, best_epoch)
        print('학습 그래프 생성 완료....')

        # best model
        model.load_state_dict(best_model)
        torch.save(model.state_dict(), self.save_path)

        print('혼동행렬 생성 중....')
        self._confusion_matrix(model, valid_loader, test_loader, device)
        print('혼동행렬 생성 완료....')

        print('최종평가 수행 중....')
        test_loss, test_acc = self._evaluation(model, test_loader, criterion, device)

        print(f'- Best_Epoch : {best_epoch + 1}')
        print(f'- Valid_Loss : {low_loss:.3f}  Valid_Accuracy : {high_acc*100:.2f}%')
        print(f'- Test_Loss : {test_loss:.3f}  Test_Accuracy : {test_acc*100:.2f}%')

        epoch_df = pd.DataFrame({'Train_Loss' : train_loss_list,'Validation_Loss' : val_loss_list, 'Train_Accuracy' : train_acc_list, 'Valid_Accuracy' : val_acc_list})
        epoch_df.index = [i + 1 for i in range(len(train_loss_list))]
        epoch_df.index.name = 'Epoch'

        result_df = pd.DataFrame({'Loss':[low_loss, test_loss], 'Accuracy':[high_acc, test_acc]})
        result_df.index = ['Validation','Test']
        result_df.index.name = 'Phase'

        with pd.ExcelWriter('result_excel.xlsx') as writer:
            epoch_df.to_excel(writer, sheet_name = 'Epoch')
            result_df.to_excel(writer, sheet_name = 'Result')

        print('학습과 평가가 종료되었습니다!!!!')
