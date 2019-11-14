#include "../../../tensor/XGlobal.h"
#include "../../../tensor/XTensor.h"
#include "../../../tensor/core/CHeader.h"
using namespace nts;

namespace xorOperate
{
	struct XOROperateModel
	{
		XTensor weight1;

		XTensor weight2;

		XTensor b1;

		XTensor b2;

		int n_hidden;

		int devID;   //  -1£∫CPU  0:0∫≈œ‘ø®
	};
	
	struct XOROperateNet
	{

		XTensor hidden_state1;

		XTensor hidden_state2;

		XTensor hidden_state3;

		XTensor hidden_state4;

		XTensor hidden_state5;

		XTensor output;
	};
	int XOROperateMain(int argc, const char ** argv);
}