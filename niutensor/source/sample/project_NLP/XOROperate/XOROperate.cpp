#include <iostream>
#include "XOROperate.h"


#include "../../../tensor/function/FHeader.h"

namespace xorOperate
{

	float learningRate = 0.01F;            // learning rate
	int nEpoch = 100;                     // 训练次数
	/*todo 修改损失函数方法*/
	float minmax = 0.01F;                 // range [-p,p] for parameter initialization
	
	void Init(XOROperateModel &model);		  //初始化参数
	void InitGrad(XOROperateModel &model, XOROperateModel &grad);
	void Forword(XTensor &input, XOROperateModel &model, XOROperateNet &net);
	void Update(XOROperateModel &model, XOROperateModel &grad, float learningRate);
	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss);
	void MSELossBackword(XTensor &output, XTensor &gold, XTensor &grad);
	void Backward(XTensor &input, XTensor &gold, XOROperateModel &model, XOROperateModel &grad, XOROperateNet &net);
	void CleanGrad(XOROperateModel &grad);


	int XOROperateMain(int argc, const char ** argv)    //项目主函数入口，传入Main.cpp
	{
		XOROperateModel model;
		model.n_hidden = 20;
		const int dataSize = 64;
		const int testDataSize = 5;
		model.devID = 0;
		Init(model);

		//准备数据集
		float dic_data[8][3] = { {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1} };

		float trainDataX[dataSize][6] = {};
		for (int i = 0; i < 8; ++i)
		{
			for (int j = 0; j < 8; ++j)
			{
				trainDataX[i * 8 + j][0] = dic_data[i][0];		//前三位为i
				trainDataX[i * 8 + j][1] = dic_data[i][1];				
				trainDataX[i * 8 + j][2] = dic_data[i][2];
				trainDataX[i * 8 + j][3] = dic_data[j][0];
				trainDataX[i * 8 + j][4] = dic_data[j][1];
				trainDataX[i * 8 + j][5] = dic_data[j][2];		// 后三位为j
			}
		}
	

		float trainDataY[dataSize][8] = {};
		for (int i = 0; i < 64; ++i)
		{
			for (int j = 0; j < 8; ++j)
				trainDataY[i][j] = 0;
		}
		float dataY[dataSize][3] = {};
		for (int i = 0; i < dataSize; ++i)
		{
			dataY[i][0] = (trainDataX[i][0] == trainDataX[i][3]) ? 0 : 1;
			dataY[i][1] = (trainDataX[i][1] == trainDataX[i][4]) ? 0 : 1;
			dataY[i][2] = (trainDataX[i][2] == trainDataX[i][5]) ? 0 : 1;
		}
		for (int i = 0; i < dataSize; ++i)
		{
			if (dataY[i][0] == 0 && dataY[i][1] == 0 && dataY[i][2] == 0)
				trainDataY[i][0] = 1;
			else if(dataY[i][0] == 0 && dataY[i][1] == 0 && dataY[i][2] == 1)
				trainDataY[i][1] = 1;
			else if (dataY[i][0] == 0 && dataY[i][1] == 1 && dataY[i][2] == 0)
				trainDataY[i][2] = 1;
			else if (dataY[i][0] == 0 && dataY[i][1] == 1 && dataY[i][2] == 1)
				trainDataY[i][3] = 1;
			else if (dataY[i][0] == 1 && dataY[i][1] == 0 && dataY[i][2] == 0)
				trainDataY[i][4] = 1;
			else if (dataY[i][0] == 1 && dataY[i][1] == 0 && dataY[i][2] == 1)
				trainDataY[i][5] = 1;
			else if (dataY[i][0] == 1 && dataY[i][1] == 1 && dataY[i][2] == 0)
				trainDataY[i][6] = 1;
			else if (dataY[i][0] == 1 && dataY[i][1] == 1 && dataY[i][2] == 1)
				trainDataY[i][7] = 1;
		}

		//训练模型
		printf("prepare data for train:\n");
		TensorList inputList;
		TensorList goldList;
		for (int i = 0; i < dataSize; ++i)
		{
			XTensor*  inputData = NewTensor2D(1, 6, X_FLOAT, model.devID);
			for (int j = 0; j < 6; ++j)
			{
				inputData->Set2D(trainDataX[i][j], 0, j);
			}
			inputList.Add(inputData);
		}
		for (int i = 0; i < dataSize; ++i)
		{
			XTensor*  goldData = NewTensor2D(1, 8, X_FLOAT, model.devID);
			for (int j = 0; j < 8; ++j)
			{
				goldData->Set2D(trainDataY[i][j], 0, j);
			}
			goldList.Add(goldData);
		}
		std::cout << "输入张量个数：" << inputList.count << std::endl;
		std::cout << "输出张量个数：" << goldList.count << std::endl;

		printf("start train\n");
		XOROperateNet net;
		XOROperateModel grad;
		InitGrad(model, grad);

		for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)	//循环训练
		{
			printf("epoch %d\n", epochIndex);
			float totalLoss = 0;
			if ((epochIndex + 1) % 50 == 0)			
				learningRate /= 3;
			for (int i = 0; i < inputList.count; ++i)
			{
				XTensor *input = inputList.GetItem(i);
				XTensor *gold = goldList.GetItem(i);
				Forword(*input, model, net);
				XTensor loss;
				MSELoss(net.output, *gold, loss);	//计算误差
				totalLoss += loss.Get1D(0);

				Backward(*input, *gold, model, grad, net);

				Update(model, grad, learningRate);

				CleanGrad(grad);

			}

			printf("loss %f\n", totalLoss / inputList.count);

		}


		float testDataX[testDataSize][6] = { {0,0,0,0,0,0} ,
											 {0,0,1,0,1,0} ,
											 {0,1,1,1,0,1} ,
											 {1,1,1,1,1,0} ,
											 {1,1,1,0,0,0} }; //测试集

		float testDataY[testDataSize][6] = { {0,0,0} ,
											 {0,1,1} ,
											 {1,1,0} ,
											 {0,0,1} ,
											 {1,1,1} }; //测试集


		int num_result = 0;
		for (int i = 0; i < testDataSize; ++i)
		{
			XTensor*  inputTestData = NewTensor2D(1, 6, X_FLOAT, model.devID);

			for (int j = 0; j < 6; ++j)
			{
				inputTestData->Set2D(testDataX[i][j], 0, j);
			}
			Forword(*inputTestData, model, net);
			printf("%f,%f,%f\n",net.output.Get2D(0, 0), net.output.Get2D(0, 1), net.output.Get2D(0, 2));

			/*
			bool result1 = (((net.output.Get2D(0, 0) - testDataY[i][0]) < 0.5 )|| (testDataY[i][0] - (net.output.Get2D(0, 0)) < 0.5)) ? true:false;
			bool result2 = (((net.output.Get2D(0, 1) - testDataY[i][1]) < 0.5) || (testDataY[i][1] - (net.output.Get2D(0, 1)) < 0.5)) ? true : false;
			bool result3 = (((net.output.Get2D(0, 2) - testDataY[i][2]) < 0.5) || (testDataY[i][2] - (net.output.Get2D(0, 2)) < 0.5)) ? true : false;
			if (result1 && result2 && result3)
			{
				num_result++;
			}
			*/
		}


		return 0;

	}

	void Init(XOROperateModel &model)
	{
		InitTensor2D(&model.weight1, 6, model.n_hidden, X_FLOAT, model.devID);		//初始化模型中的tensor  w1
		InitTensor2D(&model.weight2, model.n_hidden, 8, X_FLOAT, model.devID);		//初始化模型中的tensor  w2
		InitTensor2D(&model.b1, 1, model.n_hidden, X_FLOAT, model.devID);			//初始化模型中的tensor  b1
		InitTensor2D(&model.b2, 1, 8, X_FLOAT, model.devID);			//初始化模型中的tensor  b2
		
		// todo 使用softmax损失函数
		
		model.weight1.SetDataRand(-minmax, minmax);		
		model.weight2.SetDataRand(-minmax, minmax);
		model.b1.SetZeroAll();	
		model.b2.SetZeroAll();
		
		printf("Initialization  complete.\n");
	}
	void InitGrad(XOROperateModel &model, XOROperateModel &grad)	
	{
		InitTensor(&grad.weight1, &model.weight1);
		InitTensor(&grad.weight2, &model.weight2);
		InitTensor(&grad.b1, &model.b1);
		InitTensor(&grad.b2, &model.b2);

		grad.n_hidden = model.n_hidden;
		grad.devID = model.devID;
	}

	void Forword(XTensor &input, XOROperateModel &model, XOROperateNet &net)
	{
		net.hidden_state1 = MatrixMul(input, model.weight1);	
		net.hidden_state2 = net.hidden_state1 + model.b1;	
		net.hidden_state3 = Rectify(net.hidden_state2);
		net.hidden_state4 = MatrixMul(net.hidden_state3, model.weight2);
		net.hidden_state5 = net.hidden_state4 + model.b2;
		net.output = Rectify(net.hidden_state5);
	}

	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss)		//计算损失
	{
		/*
		XTensor tmp = output - gold;
		loss = ReduceSum(tmp, 1, 2) / output.dimSize[1];
		*/
		loss = CrossEntropy(output, gold);
	}

	void MSELossBackword(XTensor &output, XTensor &gold, XTensor &grad)		
	{
		/*
		_CrossEntropyBackward(&grad, &output, &gold);
		*/

		grad = CrossEntropy(output,gold);

		/*
		XTensor tmp = output - gold;
		grad = tmp * 2;
		*/
	}

	void Backward(XTensor &input, XTensor &gold, XOROperateModel &model, XOROperateModel &grad, XOROperateNet &net) 
	{
		XTensor lossGrad;
		XTensor &dedb2 = grad.b2;
		XTensor &dedw2 = grad.weight2;
		XTensor &dedb1 = grad.b1;
		XTensor &dedw1 = grad.weight1;


		MSELossBackword(net.output, gold, lossGrad);

		_RectifyBackward(&net.output, &net.hidden_state5, &lossGrad, &dedb2);

		MatrixMul(net.hidden_state3, X_TRANS, dedb2, X_NOTRANS, dedw2);

		XTensor dedy2 = MatrixMul(dedb2, X_NOTRANS, model.weight2, X_TRANS);

		_RectifyBackward(&net.hidden_state3, &net.hidden_state2, &dedy2, &dedb1);

		MatrixMul(input, X_TRANS, dedb1, X_NOTRANS, dedw1);

	}

	void Update(XOROperateModel &model, XOROperateModel &grad, float learningRate)    //更新训练模型
	{

		model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);

		model.weight2 = Sum(model.weight2, grad.weight2, -learningRate);

		model.b1 = Sum(model.b1, grad.b1, -learningRate);

		model.b2 = Sum(model.b2, grad.b2, -learningRate);
	}

	void CleanGrad(XOROperateModel &grad)	//清空grad的w1，w2和b
	{
		grad.b1.SetZeroAll();
		grad.b2.SetZeroAll();
		grad.weight1.SetZeroAll();
		grad.weight2.SetZeroAll();
	}

}