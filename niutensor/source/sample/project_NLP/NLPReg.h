#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"
using namespace nts;

namespace project_NLP
{
	struct NLPRegModel   // 回归模型类
	{
		XTensor weight1;   // 第一层权重

		XTensor weight2;   // 第二层权重

		XTensor b;

		int h_size;

		int devID;   //  -1：CPU  0:0号显卡
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