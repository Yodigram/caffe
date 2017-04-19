#include <vector>
#include <omp.h>
#include <cmath>
#include <caffe/caffe.hpp>

#include "caffe/blob.hpp"
#include "caffe/yodi_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	//==================================================================
	/* Example prototxt
	 *
		layer {
		  name: "quantize"
		  type: "Quantize"
		  bottom: "data"
		  top: "data"
		  quantize_param {
			min : -1.0
			max : 1.0
			bins : 8
		  }
		  include {
			phase: TRAIN
		  }
		}
	*/
	//==================================================================

	template<typename Dtype>
	void QuantizeLayer<Dtype>::LayerSetUp(
			const vector<Blob<Dtype> *>& bottom,
			const vector<Blob<Dtype> *>& top)
	{
		QuantizeParameter params =
				this->layer_param_.quantize_param();
	    //----------------------------------------
	    if (params.has_min())
	    {
	    	m_min = Dtype(params.min());
	    }

	    if (params.has_max())
	    {
	    	m_max = Dtype(params.max());
	    }

	    CHECK_GT(m_max, m_min) << "max must be > min.";
	    //----------------------------------------
	    if (params.has_bins())
	    {
	    	m_bins = int(params.bins());
	    }

	    CHECK_GT(m_bins, 1) << "bins must be > 1";
	    //----------------------------------------
		m_div = (m_max - m_min) / m_bins;
	}

	//==================================================================

	template<typename Dtype>
	void QuantizeLayer<Dtype>::Reshape(
			const vector<Blob<Dtype> *>& bottom,
			const vector<Blob<Dtype> *>& top)
	{
		NeuronLayer<Dtype>::Reshape(bottom, top);
	}

	//==================================================================

	template<typename Dtype>
	void QuantizeLayer<Dtype>::Forward_cpu(
			const vector<Blob<Dtype> *>& bottom,
			const vector<Blob<Dtype> *>& top)
	{
		Blob<Dtype>* bottomBlob = bottom[0];
		const Dtype* bottomData = bottomBlob->cpu_data();
		Blob<Dtype>* topBlob = top[0];
		Dtype* topData = topBlob->mutable_cpu_data();

		//--------- iterate over images in blob
		const int count = bottomBlob->count();

		#pragma omp parallel for
		for (int i = 0; i < count; ++i)
		{
			Dtype v = bottomData[i];

			if (v >= m_max)
			{
				topData[i] = m_max;
			}
			else if (v <= m_min)
			{
				topData[i] = m_min;
			}
			else
			{
				topData[i] = round(v / m_div) * m_div;
			}
		}
	}

	//==================================================================

	template<typename Dtype>
	void QuantizeLayer<Dtype>::Backward_cpu(
			const vector<Blob<Dtype> *>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *>& bottom)
	{
		if (!propagate_down[0])
		{
			return;
		}

		if (top[0] == bottom[0])
		{
			return;
		}

		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		caffe_copy(top[0]->count(), top_diff, bottom_diff);
	}

	//==================================================================
	
	INSTANTIATE_CLASS(QuantizeLayer);
	REGISTER_LAYER_CLASS(Quantize);
};
