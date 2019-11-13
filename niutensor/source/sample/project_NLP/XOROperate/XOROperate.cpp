#include <iostream>

#include "../../../tensor/function/FHeader.h"

namespace project_NLP
{
	//ģ����6�����룬3�����
	
	int XOROperateMain(int argc, const char ** argv)    //��Ŀ��������ڣ�����Main.cpp
	{
		const int dataSize = 64;
		const int testDataSize = 5;

		float dic_data[8][3] = { {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1} };

		float trainDataX[dataSize][6] = {};
		for (int i = 0; i < 8; ++i)
		{
			for (int j = 0; j < 8; ++j)
			{
				trainDataX[i * 8 + j][0] = dic_data[i][0];		//ǰ��λΪi
				trainDataX[i * 8 + j][1] = dic_data[i][1];				
				trainDataX[i * 8 + j][2] = dic_data[i][2];
				trainDataX[i * 8 + j][3] = dic_data[j][0];
				trainDataX[i * 8 + j][4] = dic_data[j][1];
				trainDataX[i * 8 + j][5] = dic_data[j][2];// ����λΪj
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

		//todo ѵ��ģ��

		float testDataX[testDataSize][6] = { {0,0,0,0,0,0} ,
											 {0,0,1,0,1,0} ,
											 {0,1,1,1,0,1} ,
											 {1,1,1,1,1,0} ,
											 {1,1,1,0,0,0} }; //���Լ�


		//todo ����ģ��׼ȷ��


		return 0;
	}


}