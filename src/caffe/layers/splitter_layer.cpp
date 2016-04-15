#include <vector>
#include <omp.h>
#include <caffe/caffe.hpp>

#include "caffe/blob.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe
{
	template<typename Dtype>
	void SplitterLayer<Dtype>::LayerSetUp(
			const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top)
	{
	}

	template<typename Dtype>
	void SplitterLayer<Dtype>::Reshape(
			const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top)
	{
		const int nb0 = bottom[0]->num();
		const int wb0 = bottom[0]->width();
		const int hb0 = bottom[0]->height();
		const int cb0 = bottom[0]->channels();

		int topSize = (int) top.size();
		CHECK(wb0 % topSize == 0);

		vector<int> newShape;
		newShape.push_back(nb0);
		newShape.push_back(cb0);
		newShape.push_back(hb0);
		newShape.push_back(wb0 / topSize);

		// Reshape all tops
		for (int i = 0; i < topSize; i++)
		{
			top[i]->Reshape(newShape);
		}
	}


	template<typename Dtype>
	void SplitterLayer<Dtype>::Forward_cpu(
			const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top)
	{
		const Blob<Dtype> *imBlob = bottom[0];
		const int         n1      = imBlob->num();
		const int         wb1     = imBlob->width();
		const int         hb1     = imBlob->height();
		const int         cb1     = imBlob->channels();

		const Dtype *bottomDataImage = imBlob->cpu_data();

		int topSize    = (int) top.size();
		int splitPoint = wb1 / topSize;

		#pragma omp parallel for
		for (int n = 0; n < n1; n++)
		{
			for (int c = 0; c < cb1; c++)
			{
				for (int h = 0; h < hb1; h++)
				{
					for (int t = 0; t < topSize; t++)
					{
						int         w_offset = t * splitPoint;
						Blob<Dtype> *outBlob = top[t];
						Dtype       *outData = outBlob->mutable_cpu_data();

						for (int w = 0; w < splitPoint; w++)
						{
							outData[outBlob->offset(n, c, h, w)] =
									bottomDataImage[imBlob->offset(n, c, h, w + w_offset)];
						}
					}
				}
			}
		}
	}

	template<typename Dtype>
	void SplitterLayer<Dtype>::Backward_cpu(
			const vector<Blob<Dtype> *> &top,
			const vector<bool> &propagate_down,
			const vector<Blob<Dtype> *> &bottom)
	{

	}

	INSTANTIATE_CLASS(SplitterLayer);
	REGISTER_LAYER_CLASS(Splitter);
}