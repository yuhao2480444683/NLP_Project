#include "../../../tensor/XGlobal.h"
#include "../../../tensor/XTensor.h"
#include "../../../tensor/core/CHeader.h"
using namespace nts;

namespace project_NLP
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
	

	int XOROperateMain(int argc, const char ** argv);
}