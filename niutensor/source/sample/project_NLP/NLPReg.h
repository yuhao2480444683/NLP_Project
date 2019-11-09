#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"
using namespace nts;

namespace project_NLP
{
	struct NLPRegModel   // �ع�ģ����
	{
		XTensor weight1;   // ��һ��Ȩ��

		XTensor weight2;   // �ڶ���Ȩ��

		XTensor b;

		int h_size;

		int devID;   //  -1��CPU  0:0���Կ�
	};

	struct NLPRegNet
	{
		/*before bias*/
		XTensor hidden_state1;  
		/*before active function*/
		XTensor hidden_state2;
		/*after active function*/
		XTensor hidden_state3;
		/*output*/
		XTensor output;
	};
	int NLPRegMain(int argc, const char ** argv);

};