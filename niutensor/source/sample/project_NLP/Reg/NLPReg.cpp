#include <iostream>
#include "NLPReg.h"

//δ��ͷ�ļ�����
#include "../../../tensor/function/FHeader.h"

namespace nlpreg
{

	float learningRate = 0.3F;            // learning rate
	int nEpoch = 100;                     // ѵ������
	float minmax = 0.01F;                 // range [-p,p] for parameter initialization

	void Init(NLPRegModel &model);		  //��ʼ������
	void InitGrad(NLPRegModel &model, NLPRegModel &grad);	
	void Train(float *trainDataX, float *trainDataY, int dataSize, NLPRegModel &model);
	void Forword(XTensor &input, NLPRegModel &model, NLPRegNet &net);
	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss);
	void Backward(XTensor &input, XTensor &gold, NLPRegModel &model, NLPRegModel &grad, NLPRegNet &net);
	void Update(NLPRegModel &model, NLPRegModel &grad, float learningRate);
	void CleanGrad(NLPRegModel &grad);
	void Test(float *testData, int testDataSize, NLPRegModel &model);


	int NLPRegMain(int argc, const char ** argv)    //��Ŀ��������ڣ�����Main.cpp
	{
		NLPRegModel model;
		model.h_size = 4;           // ���ز�ڵ����(��ȣ�)
		const int dataSize = 16;
		const int testDataSize = 3;
		model.devID = 0;			// -1��������cpu  0:0���Կ�
		Init(model);				//��ʼ������

		/*train Data*/
		float trainDataX[dataSize] = { 51,56.8,58,63,66,69,73,76,81,85,90,94,97,100,103,107 };	//ѵ����X
		float trainDataY[dataSize] = { 31,34.7,35.6,36.7,39.5,42,42.7,47,49,51,52.5,54,55.7,56,58.8,59.2 };	//ѵ����Y

		float testDataX[testDataSize] = { 64, 80, 95 };		//���Լ�

		Train(trainDataX, trainDataY, dataSize, model);		//ѵ���ع�ģ��

		Test(testDataX, testDataSize, model);				//ʹ��ģ�ͽ���Ԥ��
		return 0;
	}

	void Init(NLPRegModel &model)
	{
		InitTensor2D(&model.weight1, 1, model.h_size, X_FLOAT, model.devID);	//��ʼ��ģ���е�tensor  w1
		InitTensor2D(&model.weight2, model.h_size, 1, X_FLOAT, model.devID);	//��ʼ��ģ���е�tensor  w2
		InitTensor2D(&model.b, model.h_size, 1, X_FLOAT, model.devID);			//��ʼ��ģ���е�tensor  b
		model.weight1.SetDataRand(-minmax, minmax);		//���÷�ΧΪ��-0.01��0.01��
		model.weight2.SetDataRand(-minmax, minmax);
		model.b.SetZeroAll();		//����ֵȫ����ʼ��Ϊ0
		printf("Initialization  complete.\n");
	}

	void InitGrad(NLPRegModel &model, NLPRegModel &grad)	//��ֵ�ȣ�
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
			XTensor*  inputData = NewTensor2D(1, 1, X_FLOAT, model.devID);	//����ֵΪһά
			inputData->Set2D(trainDataX[i] / 100, 0, 0);					//��100���ҵ�����ǹ�һ����ʹֵ�򱣳���0-1֮��
			inputList.Add(inputData);		//�����б���뵱ǰֵ

			XTensor*  goldData = NewTensor2D(1, 1, X_FLOAT, model.devID);	//��������
			goldData->Set2D(trainDataY[i] / 60, 0, 0);						//����ֵ/60����������ݱ������
			goldList.Add(goldData);			//�����б���뵱ǰֵ
		}

		printf("start train\n");
		NLPRegNet net;
		NLPRegModel grad;
		InitGrad(model, grad);		//main��ʼʱ�Զ����ģ�Ͳ�������model��ʼ��grad
		for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)	//ѭ��ѵ��
		{
			printf("epoch %d\n", epochIndex);
			float totalLoss = 0;
			if ((epochIndex + 1) % 50 == 0)			//�����ѵ�����������˴���50��֮��ѧϰ�ʳ�3��
				learningRate /= 3;
			for (int i = 0; i < inputList.count; ++i)	
			{
				XTensor *input = inputList.GetItem(i);
				XTensor *gold = goldList.GetItem(i);
				Forword(*input, model, net);	//���򴫲�
				//output.Dump(stderr);
				XTensor loss;
				MSELoss(net.output, *gold, loss);	//�������
				//loss.Dump(stderr);
				totalLoss += loss.Get1D(0);		//�����

				Backward(*input, *gold, model, grad, net);	//����

				Update(model, grad, learningRate);	

				CleanGrad(grad);	//������һ����׼��

			}
			printf("loss %f\n", totalLoss / inputList.count);
		}
	}
	void Forword(XTensor &input, NLPRegModel &model, NLPRegNet &net)
	{
		net.hidden_state1 = MatrixMul(input, model.weight1);		//��һ�����
		net.hidden_state2 = net.hidden_state1 + model.b;			//��һ�����ƫ��
		net.hidden_state3 = HardTanH(net.hidden_state2);			//��һ��û����
		net.output = MatrixMul(net.hidden_state3, model.weight2);	//�ڶ������״̬ת�������Ԥ��ֵ
	}

	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss)		//������ʧ
	{
		XTensor tmp = output - gold;
		loss = ReduceSum(tmp, 1, 2) / output.dimSize[1];
	}

	void MSELossBackword(XTensor &output, XTensor &gold, XTensor &grad)		//������ʧ���Ƕ�����ϵô...
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

	void Update(NLPRegModel &model, NLPRegModel &grad, float learningRate)    //����ѵ��ģ��
	{
		model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);	//��һ�ε�Ȩ�ؼӱ���ѵ����Ȩ�أ�������ѵ�����ѧϰ��
		model.weight2 = Sum(model.weight2, grad.weight2, -learningRate);
		model.b = Sum(model.b, grad.b, -learningRate);
	}

	void CleanGrad(NLPRegModel &grad)	//���grad��w1��w2��b
	{
		grad.b.SetZeroAll();
		grad.weight1.SetZeroAll();
		grad.weight2.SetZeroAll();
	}

	void Test(float *testData, int testDataSize, NLPRegModel &model)	//ʹ��ģ�ͽ���Ԥ��
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