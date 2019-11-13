#include <iostream>
#include "XOROperate.h"

#include "../../../tensor/function/FHeader.h"

namespace project_NLP
{
	//模型有6个输入，3个输出
	
	int XOROperateMain(int argc, const char ** argv)    //项目主函数入口，传入Main.cpp
	{
		const int dataSize = 64;
		const int testDataSize = 5;

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
				trainDataX[i * 8 + j][5] = dic_data[j][2];// 后三位为j
			}
		}
		std::cout << "trainDataX : ";
		for (int i = 0; i < 64; ++i)
		{
			std::cout << trainDataX[i][0] << trainDataX[i][1] << trainDataX[i][2] << trainDataX[i][3] << trainDataX[i][4] << trainDataX[i][5] << std::endl;
		}
		
		float trainDataY[dataSize][3] = {};
		for (int i = 0; i < dataSize; ++i)
		{
			trainDataY[i][0] = (trainDataX[i][0] == trainDataX[i][3]) ? 0 : 1;
			trainDataY[i][1] = (trainDataX[i][1] == trainDataX[i][4]) ? 0 : 1;
			trainDataY[i][2] = (trainDataX[i][2] == trainDataX[i][5]) ? 0 : 1;
		}
		std::cout << "trainDataY : ";
		for (int i = 0; i < 64; ++i)
		{
			std::cout << trainDataY[i][0] << trainDataY[i][1] << trainDataY[i][2] << std::endl;
		}

		
		//初始化模型，给定参数
		//训练模型



		

		float testDataX[testDataSize][6] = { {0,0,0,0,0,0} ,
											 {0,0,1,0,1,0} ,
											 {0,1,1,1,0,1} ,
											 {1,1,1,1,1,0} ,
											 {1,1,1,0,0,0} }; //测试集


		//todo 测试模型准确率


		return 0;
	}

	void Init(XOROperateModel &model)
	{
		InitTensor2D(&model.weight1, 1, model.n_hidden, X_FLOAT, model.devID);		//初始化模型中的tensor  w1
		InitTensor2D(&model.weight2, model.n_hidden, 1, X_FLOAT, model.devID);		//初始化模型中的tensor  w2
		InitTensor2D(&model.b1, model.n_hidden, 1, X_FLOAT, model.devID);			//初始化模型中的tensor  b1
		InitTensor2D(&model.b2, model.n_hidden, 1, X_FLOAT, model.devID);			//初始化模型中的tensor  b2
		
		// todo 使用softmax损失函数
		/*
		model.weight1.SetDataRand(-minmax, minmax);		
		model.weight2.SetDataRand(-minmax, minmax);
		model.b.SetZeroAll();	
		*/
		printf("Initialization  complete.\n");
	}

}