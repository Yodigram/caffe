#include <vector>
#include <omp.h>
#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

#include "caffe/blob.hpp"

namespace caffe
{

	template<typename Dtype>
	void UnwarpLayer<Dtype>::LayerSetUp(
			const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top)
	{
		_maxStep      = 0;
		_blockSize    = 0;
		_defaultValue = 0;
	}

	//------------------------------------------------------------------------------------------------------------------

	template<typename Dtype>
	void UnwarpLayer<Dtype>::Reshape(
			const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top)
	{
		const int nb0 = bottom[0]->num();
		const int wb0 = bottom[0]->width();
		const int hb0 = bottom[0]->height();
		const int cb0 = bottom[0]->channels();

		const int nb1 = bottom[1]->num();
		const int wb1 = bottom[1]->width();
		const int hb1 = bottom[1]->height();
		const int cb1 = bottom[1]->channels();
/*
		LOG(INFO) << "wb0=" << wb0 <<", hb0=" << hb0 <<", cb0="<< cb0 << " nb0=" << nb0;
		LOG(INFO) << "wb1=" << wb1 <<", hb1=" << hb1 <<", cb1="<< cb1 << " nb1=" << nb1;
*/
		CHECK(hb0 == 1);
		CHECK(wb0 == 1);


		//CHECK(cb0 == (wb1 * hb1 * 2));
		CHECK(wb1 == hb1);
		CHECK(cb0 >= 2 && cb0 % 2 == 0);

		_noBlocks  = int(std::sqrt(cb0 / 2));
		_blockSize = (int)std::ceil(float(wb1) / float(_noBlocks));
		_maxStep   = std::max(hb1 * 0.2, wb1 * 0.2);

		// Reshape all tops
		int topSize = (int) top.size();

		// if we have one distortion we need both
		bool haveX = topSize > 1;
		bool haveY = topSize > 2;
		if (haveX)
		{
			CHECK(haveX && haveY);
		}

		for (int i = 0; i < topSize; i++)
		{
			top[i]->ReshapeLike(*bottom[1]);
		}


		// Update debug mats
		#ifdef USE_OPENCV
		cv_img_wrap = cv::Mat(hb1, wb1, CV_8UC3);
		cv_img_unwrap = cv::Mat(hb1, wb1, CV_8UC3);
		cv_img_dist_h = cv::Mat(hb1, wb1, CV_8UC3);
		cv_img_dist_w = cv::Mat(hb1, wb1, CV_8UC3);
		cv_img_map = cv::Mat(hb1, wb1, CV_8UC3);
		#endif
	}

	//------------------------------------------------------------------------------------------------------------------

	template<typename Dtype>
	float UnwarpLayer<Dtype>::getTH(int h, int w)
	{
		return (std::floor(float(h) / float(_blockSize)) * float(_noBlocks) +
				std::floor(float(w) / float(_blockSize))) * 2;
	}

	//------------------------------------------------------------------------------------------------------------------

	template<typename Dtype>
	void UnwarpLayer<Dtype>::Forward_cpu(
			const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top)
	{
		bool haveDistMap = top.size() > 2;


		const Blob<Dtype> *mapBlob = bottom[0];
		const Blob<Dtype> *imBlob  = bottom[1];

		const int n1 = imBlob->num();
		const int w1 = imBlob->width();
		const int h1 = imBlob->height();
		const int c1 = imBlob->channels();

		const int nb0 = bottom[0]->num();
		const int wb0 = bottom[0]->width();
		const int hb0 = bottom[0]->height();
		const int cb0 = bottom[0]->channels();

		const int   nb1           = bottom[1]->num();
		const int   wb1           = bottom[1]->width();
		const int   hb1           = bottom[1]->height();
		const int   cb1           = bottom[1]->channels();
/*
		LOG(INFO) << "wb0=" << wb0 <<", hb0=" << hb0 <<", cb0="<< cb0 << " nb0=" << nb0;
		LOG(INFO) << "wb1=" << wb1 <<", hb1=" << hb1 <<", cb1="<< cb1 << " nb1=" << nb1;
		LOG(INFO) << "w1="  << w1  <<", h1="  << h1  <<", c1=" << c1  << " n1="  << n1;
*/
		Blob<Dtype> *outBlobImage = top[0];
		Dtype       *outDataImage = outBlobImage->mutable_cpu_data();
		Blob<Dtype> *outBlobH     = NULL;
		Dtype       *outDataH     = NULL;
		Blob<Dtype> *outBlobW     = NULL;
		Dtype       *outDataW     = NULL;

		if (haveDistMap)
		{
			outBlobH = top[1];
			outDataH = outBlobH->mutable_cpu_data();
			outBlobW = top[2];
			outDataW = outBlobW->mutable_cpu_data();
		}

		const Dtype *bottomDataMap   = mapBlob->cpu_data();
		const Dtype *bottomDataImage = imBlob->cpu_data();

		Dtype maxMapPixel(-999999999999);
		Dtype minMapPixel(999999999999);

		#pragma omp parallel for
		for (int n = 0; n < n1; n++)
		{
			for (int h = 0; h < h1; h++)
			{
				for (int w = 0; w < w1; w++)
				{
					// find h,w map of offset
					const int t_h = (int) (getTH(h, w));
					const int t_w = t_h + 1;


					// Debug
					{
						Dtype val = bottomDataMap[mapBlob->offset(n, t_h, 0, 0)];
						if (maxMapPixel < val) { maxMapPixel = val; }
						if (minMapPixel > val) { minMapPixel = val; }
					}

					Dtype m_h = h - bottomDataMap[mapBlob->offset(n, t_h, 0, 0)] * _maxStep;
					Dtype m_w = w - bottomDataMap[mapBlob->offset(n, t_w, 0, 0)] * _maxStep;

					const int m_h_int = std::floor(m_h);
					const int m_w_int = std::floor(m_w);

					Dtype valueDistH;
					Dtype valueDistW;
					if (w < w1 - 1 && h < h1 - 1)
					{
						const int t_h_00        = (int) std::floor(getTH(h, w));
						Dtype     valueDistH_00 = (bottomDataMap[mapBlob->offset(n, t_h_00, 0, 0)] + 1) * 255 / 2;
						const int t_h_10        = (int) std::floor(getTH(h + 1, w));
						Dtype     valueDistH_10 = (bottomDataMap[mapBlob->offset(n, t_h_10, 0, 0)] + 1) * 255 / 2;
						const int t_h_01        = (int) std::floor(getTH(h, w + 1));
						Dtype     valueDistH_01 = (bottomDataMap[mapBlob->offset(n, t_h_01, 0, 0)] + 1) * 255 / 2;
						const int t_h_11        = (int) std::floor(getTH(h + 1, w + 1));
						Dtype     valueDistH_11 = (bottomDataMap[mapBlob->offset(n, t_h_11, 0, 0)] + 1) * 255 / 2;

						const float h_d = float(h % _blockSize) / float(_blockSize);
						const float w_d = float(w % _blockSize) / float(_blockSize);

						valueDistH = valueDistH_00 * (1 - h_d) * (1 - w_d) +
									 valueDistH_10 * h_d * (1 - w_d) +
									 valueDistH_01 * (1 - h_d) * w_d +
									 valueDistH_11 * w_d * h_d;

						const int t_w_00        = t_h_00 + 1;
						Dtype     valueDistW_00 = (bottomDataMap[mapBlob->offset(n, t_w_00, 0, 0)] + 1) * 255 / 2;
						const int t_w_10        = t_h_10 + 1;
						Dtype     valueDistW_10 = (bottomDataMap[mapBlob->offset(n, t_w_10, 0, 0)] + 1) * 255 / 2;
						const int t_w_01        = t_h_01 + 1;
						Dtype     valueDistW_01 = (bottomDataMap[mapBlob->offset(n, t_w_01, 0, 0)] + 1) * 255 / 2;
						const int t_w_11        = t_h_11 + 1;
						Dtype     valueDistW_11 = (bottomDataMap[mapBlob->offset(n, t_w_11, 0, 0)] + 1) * 255 / 2;

						valueDistW = valueDistW_00 * (1 - h_d) * (1 - w_d) +
									 valueDistW_10 * h_d * (1 - w_d) +
									 valueDistW_01 * (1 - h_d) * w_d +
									 valueDistW_11 * w_d * h_d;
					}
					else
					{
						valueDistH = (bottomDataMap[mapBlob->offset(n, t_h, 0, 0)] + 1) * 255 / 2;
						valueDistW = (bottomDataMap[mapBlob->offset(n, t_w, 0, 0)] + 1) * 255 / 2;
					}


					if (valueDistH > 255)
						valueDistH = 255;
					if (valueDistH < 0)
						valueDistH = 0;
					if (valueDistW > 255)
						valueDistW = 255;
					if (valueDistW < 0)
						valueDistW = 0;

					for (int c = 0; c < cb1; c++)
					{
						// Sett image
						if (m_h_int < h1 && m_h_int >= 0 &&
							m_w_int < w1 && m_w_int >= 0)
						{
							outDataImage[outBlobImage->offset(n, c, h, w)] =
									bottomDataImage[imBlob->offset(n, c, m_h_int, m_w_int)];
						}
						else
						{
							outDataImage[outBlobImage->offset(n, c, h, w)] = _defaultValue;
						}

						// Calculate dist map
						if (haveDistMap)
						{
							outDataH[outBlobH->offset(n, c, h, w)] = valueDistH;
							outDataW[outBlobW->offset(n, c, h, w)] = valueDistW;
						}
					}
				}
			}
		}

		#ifdef USE_OPENCV
		// Export test image
		if(true)
		{
			for (int h = 0; h < h1; ++h)
			{
				for (int w = 0; w < w1; ++w)
				{
					cv::Point2i p(w,h);
					cv::Vec3b color_origin(0,0,0);
					cv::Vec3b color_unwrap(0,0,0);

					cv::Vec3b color_dist_h(0,0,0);
					cv::Vec3b color_dist_w(0,0,0);

					cv::Vec3b color_map(0,0,0);

					for (int c = 0; c < cb1; ++c)
					{
						color_origin[c] = (uchar)bottomDataImage[imBlob->offset(0, c, h, w)];
						color_unwrap[c] = (uchar)outDataImage[outBlobImage->offset(0, c, h, w)];

						color_dist_h[c] = (uchar)outDataH[outBlobH->offset(0, c, h, w)];
						color_dist_w[c] = (uchar)outDataW[outBlobW->offset(0, c, h, w)];
					}

					// find h,w map of offset
					const int t_h = (int) ((std::floor(float(h) / float(_blockSize)) * float(_noBlocks) +
											std::floor(float(w) / float(_blockSize))) * 2);
					const int t_w = t_h + 1;
					color_map[0] = (uchar)((bottomDataMap[mapBlob->offset(0, t_h, 0, 0)]-minMapPixel)/(maxMapPixel-minMapPixel)*255);
					color_map[1] = (uchar)((bottomDataMap[mapBlob->offset(0, t_w, 0, 0)]-minMapPixel)/(maxMapPixel-minMapPixel)*255);
					color_map[2] = 0;

					cv_img_wrap.at<cv::Vec3b>(p) = color_origin;
					cv_img_unwrap.at<cv::Vec3b>(p) = color_unwrap;

					cv_img_dist_h.at<cv::Vec3b>(p) = color_dist_h;
					cv_img_dist_w.at<cv::Vec3b>(p) = color_dist_w;

					cv_img_map.at<cv::Vec3b>(p) = color_map;
				}
			}


			LOG(INFO) << "UnwarpLayer<Dtype>::Farward_cpu "
			<< "  minMapPixel:" << minMapPixel << "  maxMapPixel:" << maxMapPixel;
		}
        #endif
	}

	//------------------------------------------------------------------------------------------------------------------

	template<typename Dtype>
	void UnwarpLayer<Dtype>::Backward_cpu(
			const vector<Blob<Dtype> *> &top,
			const vector<bool> &propagate_down,
			const vector<Blob<Dtype> *> &bottom)
	{
		if (propagate_down[0])
		{
			bool haveDistMap = top.size() > 2;

			Blob<Dtype> *mapBlob   = bottom[0];
			Blob<Dtype> *errorBlob = top[0];
			const int   ne         = errorBlob->num();
			const int   we         = errorBlob->width();
			const int   he         = errorBlob->height();
			const int   ce         = errorBlob->channels();

			const Dtype *topDiff = errorBlob->cpu_diff();
			Dtype       *mapDiff = mapBlob->mutable_cpu_diff();

			Blob<Dtype> *outBlobH = NULL;
			Dtype       *outDataH = NULL;
			Blob<Dtype> *outBlobW = NULL;
			Dtype       *outDataW = NULL;

			if (haveDistMap)
			{
				outBlobH = top[1];
				outDataH = outBlobH->mutable_cpu_diff();
				outBlobW = top[2];
				outDataW = outBlobW->mutable_cpu_diff();
			}

			Dtype maxError(-999999999999);
			Dtype minError(999999999999);


			Dtype maxErrorH(-999999999999);
			Dtype minErrorH(999999999999);
			Dtype maxErrorW(-999999999999);
			Dtype minErrorW(999999999999);

			Dtype maxErrorPixel(-999999999999);
			Dtype minErrorPixel(999999999999);

			#pragma omp parallel for
			for (int n = 0; n < ne; n++)
			{
				for (int h = 0; h < he; h += _blockSize)
				{
					for (int w = 0; w < we; w += _blockSize)
					{
						if (haveDistMap)
						{
							// Calculate error
							Dtype errorH = 0;
							Dtype errorW = 0;

							for (int h_offset = 0; h_offset < _blockSize; h_offset++)
							{
								for (int w_offset = 0; w_offset < _blockSize; w_offset++)
								{
									for (int c = 0; c < ce; c++)
									{
										errorH += outDataH[outBlobH->offset(n, c, h + h_offset, w + w_offset)]/255;
										errorW += outDataW[outBlobW->offset(n, c, h + h_offset, w + w_offset)]/255;

										Dtype val = outDataW[outBlobW->offset(n, c, h + h_offset, w + w_offset)];
										if (maxErrorPixel < val) { maxErrorPixel = val; }
										if (minErrorPixel > val) { minErrorPixel = val; }
									}
								}
							}
							errorH /= (_blockSize * _blockSize * ce);
							errorW /= (_blockSize * _blockSize * ce);

							// Statistics
							if (minErrorH > errorH) { minErrorH = errorH; }
							if (maxErrorH < errorH) { maxErrorH = errorH; }
							if (minErrorW > errorW) { minErrorW = errorW; }
							if (maxErrorW < errorW) { maxErrorW = errorW; }

							// Mapping
							const int t_h = (int) ((std::floor(float(h) / float(_blockSize)) * float(_noBlocks) +
													std::floor(float(w) / float(_blockSize))) * 2);
							const int t_w = t_h + 1;

							mapDiff[mapBlob->offset(n, t_h, 0, 0)] = errorH;
							mapDiff[mapBlob->offset(n, t_w, 0, 0)] = errorW;
						}
						else
						{
							// Calculate error
							Dtype error = 0;

							for (int h_offset = 0; h_offset < _blockSize; h_offset++)
							{
								for (int w_offset = 0; w_offset < _blockSize; w_offset++)
								{
									for (int c = 0; c < ce; c++)
									{
										error += topDiff[errorBlob->offset(n, c, h + h_offset, w + w_offset)] / 255;
									}
								}
							}
							error /= (_blockSize * _blockSize * ce);

							// Statistics
							if (maxError < error) { maxError = error; }
							if (minError > error) { minError = error; }

							// Mapping
							const int t_h = (int) (getTH(h,w));
							const int t_w = t_h + 1;

							mapDiff[mapBlob->offset(n, t_h, 0, 0)] = error;
							mapDiff[mapBlob->offset(n, t_w, 0, 0)] = error;
						}
					}
				}
			}


			#ifdef USE_OPENCV
			// Export test image
			if(true)
			{
				// Now we are processing only the first image
				cv::Mat  cv_img_error_h(he, we, CV_8UC3);
				cv::Mat  cv_img_error_w(he, we, CV_8UC3);

				for (int h = 0; h < he; ++h)
				{
					for (int w = 0; w < we; ++w)
					{
						cv::Point2i p(w,h);
						cv::Vec3b color_dist_h(0,0,0);
						cv::Vec3b color_dist_w(0,0,0);

						for (int c = 0; c < ce; ++c)
						{
							color_dist_h[c] = (uchar) (255 * (outDataH[outBlobH->offset(0, c, h, w)] - minErrorPixel) / (maxErrorPixel - minErrorPixel));
							color_dist_w[c] = (uchar) (255 * (outDataW[outBlobW->offset(0, c, h, w)] - minErrorPixel) / (maxErrorPixel - minErrorPixel));
						}

						cv_img_error_h.at<cv::Vec3b>(p) = color_dist_h;
						cv_img_error_w.at<cv::Vec3b>(p) = color_dist_w;
					}
				}

				// Write images to hard disk
				cv::imwrite("error_h.jpeg", cv_img_error_h);
				cv::imwrite("error_w.jpeg", cv_img_error_w);

				// Write images to hard disk
				cv::imwrite("wrap.jpeg", cv_img_wrap);
				cv::imwrite("unwrap.jpeg", cv_img_unwrap);

				cv::imwrite("distmap_h.jpeg", cv_img_dist_h);
				cv::imwrite("distmap_w.jpeg", cv_img_dist_w);

				LOG(INFO)<<"Finished update errors images";
			}
            #endif

			if (haveDistMap)
			{
				LOG(INFO) << "UnwarpLayer<Dtype>::Backward_cpu "
				<< "  maxErrorH:" << maxErrorH << "  minErrorH:" << minErrorH
				<< "  maxErrorW:" << maxErrorW << "  minErrorW:" << minErrorW
				<< "  maxErrorPixel:" << maxErrorPixel << "  minErrorPixel:" << minErrorPixel;
			}
			else
			{
				LOG(INFO) << "UnwarpLayer<Dtype>::Backward_cpu maxError:"
				<< maxError << "  minError:" << minError;
			}
		}
	}

	//------------------------------------------------------------------------------------------------------------------

	INSTANTIATE_CLASS(UnwarpLayer);
	REGISTER_LAYER_CLASS(Unwarp);
};
