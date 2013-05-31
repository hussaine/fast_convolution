/* 
 *  Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  File:    PartsBasedDetector.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 21, 2012
 */

#include "PartsBasedDetector.hpp"
#include "nms.hpp"
#include "HOGFeatures.hpp"
#include "SpatialConvolutionEngine.hpp"
#include <cstdio>
#include <iostream>

// FFLD datastructures
#include "SimpleOpt.h"
#include "Intersector.h"
#include "Mixture.h"
#include "Scene.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include "HOGPyramid.h"
#include "Patchwork.h"
#include "IFeatures.hpp"
#include "JPEGImage.h"

using namespace cv;
using namespace std;
using namespace Eigen;

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, Mat& dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        Mat _src(src.cols(), src.rows(), DataType<_Tp>::type,
              (void*)src.data(), src.stride()*sizeof(_Tp));
        transpose(_src, dst);
    }
    else
    {
        Mat _src(src.rows(), src.cols(), DataType<_Tp>::type,
                 (void*)src.data(), src.stride()*sizeof(_Tp));
        _src.copyTo(dst);
    }
}

void cv2eigen( const Mat& src,
               FFLD::HOGPyramid::Matrix & dst )
{
    dst.resize(src.rows, src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        Mat _dst(src.cols, src.rows, DataType<float>::type,
             dst.data(), (size_t)(dst.stride()*sizeof(float)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
    else
    {
        Mat _dst(src.rows, src.cols, DataType<float>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(float)));
        src.convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
}

/*! @brief search an image for potential candidates
 *
 * calls detect(const Mat& im, const Mat&depth=Mat(), vector<Candidate>& candidates);
 *
 * @param im the input color or grayscale image
 * @param candidates the output vector of detection candidates above the threshold
 */
template<typename T>
void PartsBasedDetector<T>::detect(const cv::Mat& im, vectorCandidate& candidates) {
	detect(im, Mat(), candidates);
}

/*! @brief search an image for potential object candidates
 *
 * This is the main entry point to the detection pipeline. Given an instantiated an populated model,
 * this method takes an input image, and attempts to find all instances of an object in that image.
 * The object, number of scales, detection confidence, etc are all defined through the Model.
 *
 * @param im the input color or grayscale image
 * @param depth the image depth image, used for depth consistency and search space pruning
 * @param candidates the output vector of detection candidates above the threshold
 */
template<typename T>
void PartsBasedDetector<T>::detect(const Mat& im, const Mat& depth, vectorCandidate& candidates) {

	// calculate a feature pyramid for the new image
	vectorMat pyramid;
	features_->pyramid(im, pyramid);

	// convolve the feature pyramid with the Part experts
	// to get probability density for each Part
	double t = (double)getTickCount();
	vector2DMat pdf;
	
	/////test code - Hussain
	cout << "FFLD Convolution Part Starting " << endl;
	int padding = 12;
	int interval = 10;
	vector<FFLD::HOGPyramid::Matrix> scores;
	vector<FFLD::Mixture::Indices> argmaxes;
	vector<vector<vector<FFLD::Model::Positions> > > positions;
	const string file("../tmp.jpg");
	cout << file << endl;
	FFLD::JPEGImage image(file);
	FFLD::HOGPyramid pyramidFFLD(image, padding, padding, interval);
	//FFLD::HOGPyramid BrisPyra2PyramidFFLD;
	
	if (!FFLD::Patchwork::Init((pyramidFFLD.levels()[0].rows() - padding + 15) & ~15,
							(pyramidFFLD.levels()[0].cols() - padding + 15) & ~15)) {
		cout << "\nCould not initialize the Patchwork class" << endl;
		//return -1;
	}
		
	cout << "Initialized FFTW in " << endl;
	FFLD::Mixture mixture;
	string modelffld("../model_car_final-1.2.4.txt");
	cout << modelffld << endl;
	ifstream in(modelffld.c_str(), ios::binary);
	
	if (!in.is_open()) {
		//showUsage();
		cerr << "\nInvalid model file " << modelffld << endl;
		//return -1;
	}
	in >> mixture;
	const int nbModels = mixture.models().size();
	const int nbLevels = pyramidFFLD.levels().size();
	vector<vector<FFLD::HOGPyramid::Matrix> > tmp(nbModels);


#pragma omp critical
	mixture.cacheFilters();

	const FFLD::Patchwork patchwork(pyramidFFLD);
	std::vector<FFLD::Patchwork::Filter> filterCache_;
	filterCache_ = mixture.filterCacheObj();
	vector<vector<FFLD::HOGPyramid::Matrix> > convolutions(filterCache_.size());
	patchwork.convolve(filterCache_, convolutions);///convolve patch with filters, 
	cout << " done patch work convolution " << endl;
	const int tmpnbFilters = 54, tmpnbPlanes = 12, tmpnbLevels = 41;

	//just test code to transfer convolution matrix of ffld to vector2dMat of bristow
	Mat C;
	vector2DMat pdfFFLD;
	// preallocate the output
	//const unsigned int M = features.size();//inner loop, reducing pyramind 
	//const unsigned int N = filters_.size();//outer loop
	//responses.resize(M, vectorMat(N));
	pdfFFLD.resize(tmpnbLevels, vectorMat(tmpnbFilters * tmpnbPlanes));
	//Mat D;
	for (int i = 0; i < tmpnbFilters * tmpnbPlanes; ++i) {
		const int k = i / tmpnbPlanes; // Filter index
		const int l = i % tmpnbPlanes; // Plane index
		for (int j = 0; j < tmpnbLevels; ++j) {
			FFLD::HOGPyramid::Matrix ffldResponse = convolutions[k][j];
			FFLD::HOGPyramid::Matrix tempffldResponse;
			eigen2cv(ffldResponse,C);
			//D=C;
			pdfFFLD[j][i]=C;
			//cout << "transfer i & j " << i << " " << j << endl;
			cv2eigen(C,tempffldResponse);
			cout << "ffldResponse dim " << ffldResponse.rows() << " " << ffldResponse.cols() << " C dim " << C.rows << " " << C.cols << " " << C.type()  << " " << C.channels() <<" ffld::Eigen dim " << tempffldResponse.rows() << " " << tempffldResponse.cols() <<endl;
		}
	}
	
	cout << " Starting Convolution - in PBM-detect function " << endl;
	convolution_engine_->pdf(pyramid, pdf);
	/*cout << " convolution size " << pdf.size() << endl;
	for (int pdfInd=0;pdfInd<pdf.size();pdfInd++)
		for (int pdfInd2=0;pdfInd2<pdf[1].size();pdfInd2++){
		Mat response;
		response=pdf[pdfInd][pdfInd2];
		cout << "pdf dim " << response.rows << " " << response.cols << endl;
	}*/

	cout << " End Convolution - in PBM-detect function " << endl;
	printf("Convolution time: %f\n", ((double)getTickCount() - t)/getTickFrequency());

	// use dynamic programming to predict the best detection candidates from the part responses
	vector4DMat Ix, Iy, Ik;
	vector2DMat rootv, rooti;
	t = (double)getTickCount();
	dp_.min(parts_, pdf, Ix, Iy, Ik, rootv, rooti);
	//dp_.min(parts_, pdfFFLD, Ix, Iy, Ik, rootv, rooti);
	printf("DP min time: %f\n", ((double)getTickCount() - t)/getTickFrequency());

	// suppress non-maximal candidates
	t = (double)getTickCount();
	//ssp_.nonMaxSuppression(rootv, features_->scales());
	printf("non-maxima suppression time: %f\n", ((double)getTickCount() - t)/getTickFrequency());

	// walk back down the tree to find the part locations
	t = (double)getTickCount();
	dp_.argmin(parts_, rootv, rooti, features_->scales(), Ix, Iy, Ik, candidates);
	printf("DP argmin time: %f\n", ((double)getTickCount() - t)/getTickFrequency());

	if (!depth.empty()) {
		//ssp_.filterCandidatesByDepth(parts_, candidates, depth, 0.03);
	}

}

/*! @brief Distribute the model parameters to the PartsBasedDetector classes
 *
 * @param model the monolithic model containing the deserialization of all model parameters
 */
template<typename T>
void PartsBasedDetector<T>::distributeModel(Model& model) {

	// the name of the Part detector
	name_ = model.name();

	// initialize the Feature engine
	features_.reset(new HOGFeatures<T>(model.binsize(), model.nscales(), model.flen(), model.norient()));

	//initialise the convolution engine
	convolution_engine_.reset(new SpatialConvolutionEngine(DataType<T>::type, model.flen()));

	// make sure the filters are of the correct precision for the Feature engine
	const unsigned int nfilters = model.filters().size();
	for (unsigned int n = 0; n < nfilters; ++n) {
		model.filters()[n].convertTo(model.filters()[n], DataType<T>::type);
	}
	convolution_engine_->setFilters(model.filters());

	// initialize the tree of Parts
	parts_ = Parts(model.filters(), model.filtersi(), model.def(), model.defi(), model.bias(), model.biasi(),
			model.anchors(), model.biasid(), model.filterid(), model.defid(), model.parentid());

	// initialize the dynamic program
	dp_ = DynamicProgram<T>(model.thresh());

}



// declare all specializations of the template
template class PartsBasedDetector<float>;
template class PartsBasedDetector<double>;
