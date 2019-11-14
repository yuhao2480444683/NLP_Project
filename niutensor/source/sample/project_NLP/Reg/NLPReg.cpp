#include <iostream>
#include "NLPReg.h"

//未在头文件包含
#include "../../../tensor/function/FHeader.h"

namespace nlpreg
{

	float learningRate = 0.3F;            // learning rate
	int nEpoch = 100;                     // 训练次数
	float minmax = 0.01F;                 // range [-p,p] for parameter initialization

	void Init(NLPRegModel &model);		  //初始化参数
	void InitGrad(NLPRegModel &model, NLPRegModel &grad);	
	void Train(float *trainDataX, float *trainDataY, int dataSize, NLPRegModel &model);
	void Forword(XTensor &input, NLPRegModel &model, NLPRegNet &net);
	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss);
	void Backward(XTensor &input, XTensor &gold, NLPRegModel &model, NLPRegModel &grad, NLPRegNet &net);
	void Update(NLPRegModel &model, NLPRegModel &grad, float learningRate);
	void CleanGrad(NLPRegModel &grad);
	void Test(float *testData, int testDataSize, NLPRegModel &model);


	int NLPRegMain(int argc, const char ** argv)    //项目主函数入口，传入Main.cpp
	{
		NLPRegModel model;
		model.h_size = 4;           // 隐藏层节点个数(宽度？)
		const int dataSize = 16;
		const int testDataSize = 3;
		model.devID = 0;			// -1：运行于cpu  0:0号显卡
		Init(model);				//初始化参数

		/*train Data*/
		float trainDataX[dataSize] = { 51,56.8,58,63,66,69,73,76,81,85,90,94,97,100,103,107 };	//训练集X
		float trainDataY[dataSize] = { 31,34.7,35.6,36.7,39.5,42,42.7,47,49,51,52.5,54,55.7,56,58.8,59.2 };	//训练集Y

		float testDataX[testDataSize] = { 64, 80, 95 };		//测试集

		Train(trainDataX, trainDataY, dataSize, model);		//训练回归模型

		Test(testDataX, testDataSize, model);				//使用模型进行预测
		return 0;
	}

	void Init(NLPRegModel &model)
	{
		InitTensor2D(&model.weight1, 1, model.h_size, X_FLOAT, model.devID);	//初始化模型中的tensor  w1
		InitTensor2D(&model.weight2, model.h_size, 1, X_FLOAT, model.devID);	//初始化模型中的tensor  w2
		InitTensor2D(&model.b, model.h_size, 1, X_FLOAT, model.devID);			//初始化模型中的tensor  b
		model.weight1.SetDataRand(-minmax, minmax);		//设置范围为（-0.01，0.01）
		model.weight2.SetDataRand(-minmax, minmax);
		model.b.SetZeroAll();		//修正值全部初始化为0
		printf("Initialization  complete.\n");
	}

	void InitGrad(NLPRegModel &model, NLPRegModel &grad)	//赋值等？
	{
		InitTensor(&grad.weight1, &model.weight1);
		InitTensor(&grad.weight2, &model.weight2);
		InitTensor(&grad.b, &model.b);

		grad.h_size = model.h_size;
		grad.devID = model.devID;
	}

	void Train(float *trainDataX, float *trainDataY, int dataSize, NLPRegModel &model)
	{
		printf("prepare data for train\n");
		/*prepare for train*/
		TensorList inputList;
		TensorList goldList;
		for (int i = 0; i < dataSize; ++i)
		{
			XTensor*  inputData = NewTensor2D(1, 1, X_FLOAT, model.devID);	//输入值为一维
			inputData->Set2D(trainDataX[i] / 100, 0, 0);					//除100，我的理解是归一化，使值域保持在0-1之间
			inputList.Add(inputData);		//输入列表加入当前值

			XTensor*  goldData = NewTensor2D(1, 1, X_FLOAT, model.devID);	//房价数据
			goldData->Set2D(trainDataY[i] / 60, 0, 0);						//房价值/60？这里和数据本身相关
			goldList.Add(goldData);			//输入列表加入当前值
		}

		printf("start train\n");
		NLPRegNet net;
		NLPRegModel grad;
		InitGrad(model, grad);		//main开始时自定义的模型参数，用model初始化grad
		for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)	//循环训练
		{
			printf("epoch %d\n", epochIndex);
			float totalLoss = 0;
			if ((epochIndex + 1) % 50 == 0)			//这里对训练次数进行了处理，50次之后学习率除3？
				learningRate /= 3;
			for (int i = 0; i < inputList.count; ++i)	
			{
				XTensor *input = inputList.GetItem(i);
				XTensor *gold = goldList.GetItem(i);
				Forword(*input, model, net);	//正向传播
				//output.Dump(stderr);
				XTensor loss;
				MSELoss(net.output, *gold, loss);	//计算误差
				//loss.Dump(stderr);
				totalLoss += loss.Get1D(0);		//总误差

				Backward(*input, *gold, model, grad, net);	//反馈

				Update(model, grad, learningRate);	

				CleanGrad(grad);	//进行下一次做准备

			}
			printf("loss %f\n", totalLoss / inputList.count);
		}
	}
	void Forword(XTensor &input, NLPRegModel &model, NLPRegNet &net)
	{
		net.hidden_state1 = MatrixMul(input, model.weight1);		//第一层计算
		net.hidden_state2 = net.hidden_state1 + model.b;			//第一层进行偏移
		net.hidden_state3 = HardTanH(net.hidden_state2);			//这一步没看懂
		net.output = MatrixMul(net.hidden_state3, model.weight2);	//第二层进行状态转换，输出预测值
	}

	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss)		//计算损失
	{
		XTensor tmp = output - gold;
		loss = ReduceSum(tmp, 1, 2) / output.dimSize[1];
	}

	void MSELossBackword(XTensor &output, XTensor &gold, XTensor &grad)		//反向损失？是二倍关系么...
	{
		XTensor tmp = output - gold;
		grad = tmp * 2;
	}

	void Backward(XTensor &input, XTensor &gold, NLPRegModel &model, NLPRegModel &grad, NLPRegNet &net) 
	{
		XTensor lossGrad;
		XTensor &dedw2 = grad.weight2;
		XTensor &dedb = grad.b;
		XTensor &dedw1 = grad.weight1;
		MSELossBackword(net.output, gold, lossGrad);
		MatrixMul(net.hidden_state3, X_TRANS, lossGrad, X_NOTRANS, dedw2);
		XTensor dedy = MatrixMul(lossGrad, X_NOTRANS, model.weight2, X_TRANS);
		_HardTanHBackward(&net.hidden_state3, &net.hidden_state2, &dedy, &dedb);
		dedw1 = MatrixMul(input, X_NOTRANS, dedb, X_TRANS);
	}

	void Update(NLPRegModel &model, NLPRegModel &grad, float learningRate)    //更新训练模型
	{
		model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);	//上一次的权重加本次训练的权重，最后减掉训练后的学习率
		model.weight2 = Sum(model.weight2, grad.weight2, -learningRate);
		model.b = Sum(model.b, grad.b, -learningRate);
	}

	void CleanGrad(NLPRegModel &grad)	//清空grad的w1，w2和b
	{
		grad.b.SetZeroAll();
		grad.weight1.SetZeroAll();
		grad.weight2.SetZeroAll();
	}

	void Test(float *testData, int testDataSize, NLPRegModel &model)	//使用模型进行预测
	{
		NLPRegNet net;
		XTensor*  inputData = NewTensor2D(1, 1, X_FLOAT, model.devID);
		for (int i = 0; i < testDataSize; ++i)
		{

			inputData->Set2D(testData[i] / 100, 0, 0);

			Forword(*inputData, model, net);
			float ans = net.output.Get2D(0, 0) * 60;
			printf("%f\n", ans);
		}

	}
}