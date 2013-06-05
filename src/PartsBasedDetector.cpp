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

FFLD::HOGPyramid::Level Convert(FFLD::HOGPyramid::Matrix & Mat )
{
	FFLD::HOGPyramid::Level result(Mat.rows(), Mat.cols()/32);
	int NbFeatures=32;

	for (int y = 0; y < Mat.rows(); ++y)
		for (int x = 0; x < Mat.cols()/32; ++x)
			for (int i = 0; i < NbFeatures; ++i){
				result(y, x)(i) = Mat(y,x*NbFeatures+i);
			}
			return result;
	//return Level(Mat.data()->data(),Mat.rows(),Mat.cols()/32);

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

	//step 1: import pyramid, and init patchwork
	int padding = 1;
	vector<FFLD::HOGPyramid::Level> levels(pyramid.size());
	
	cout << " Bristow pyramid size " << pyramid.size() << endl;
	int stride=32;
	vectorMat pyramidCopy;	
	for (int pyr=0;pyr<pyramid.size();pyr++){
		Mat py=pyramid[pyr];
		//cout << " py reshapes to vector Mat " << py.rows << " " << py.cols << " " << py.channels() << endl;
		///pyramid to cvmat is py, now cvmat to eigen mat
		FFLD::HOGPyramid::Matrix tmpffldMat;
		cv2eigen(py,tmpffldMat);
		//cout << "cvmat to eigen mat " << tmpffldMat.rows() << " " << tmpffldMat.cols() << endl;
		///now eigen mat to pyramid level
		levels[pyr]=Convert(tmpffldMat);
		//cout << "eigen mat to level " << levels[pyr].rows() << " " << levels[pyr].cols() << endl;


		/*Mat pyre=py.reshape(stride);
		cout << " pyre reshapes to vector Mat " << pyre.rows << " " << pyre.cols << " " << pyre.channels() << endl;
		Mat pyrere=pyre.reshape(1);
		cout << " pyre reshapes to vector Mat " << pyrere.rows << " " << pyrere.cols << " " << pyrere.channels() << endl;
		
		*/
		/*vectorMat featurev;
		split(py.reshape(stride), featurev);
		
		vectorMat tmpfeaurev;
		
		for (int pyc=0;pyc<stride;pyc++){
		cout << "pyramid size level " << pyr << " r & c " << featurev[pyc].rows << " " << featurev[pyc].cols << endl; 
		Mat tmpMat=featurev[pyc];
		}*/
	}
	// creat a ffld pyramid with bristows copy
	FFLD::HOGPyramid BrisPyra2PyramidFFLD(padding,padding,3,levels);
	cout << " pyramid specs " << BrisPyra2PyramidFFLD.levels()[0].rows() << " " << BrisPyra2PyramidFFLD.levels()[0].cols() << endl;
		if (!FFLD::Patchwork::Init((BrisPyra2PyramidFFLD.levels()[0].rows() - padding + 15) & ~15,
							(BrisPyra2PyramidFFLD.levels()[0].cols() - padding + 15) & ~15)) {
		cout << "\nCould not initialize the Patchwork class" << endl;
		//return -1;
	}
	const FFLD::Patchwork patchworkBristow(BrisPyra2PyramidFFLD);

	//step 2: transfer filters
	//no need for filter engine
	vector2DFilterEngine bristowFilters=convolution_engine_->filters();
	//convert vector2D filter to cvmat
	vector<FFLD::HOGPyramid::Level> levels4Filters(bristowFilters.size());
	for (int tempj=0;tempj<bristowFilters.size();tempj++) {
		Mat py=pdbFilters[tempj];
		//cout << " filter size " << pdbFilters.size() <<" " << py.rows << " " << py.cols << endl;
		//cout << " py reshapes to vector Mat " << py.rows << " " << py.cols << " " << py.channels() << endl;
		FFLD::HOGPyramid::Matrix tmpffldMat;
		cv2eigen(py,tmpffldMat);
		//cout << "cvmat to eigen mat " << tmpffldMat.rows() << " " << tmpffldMat.cols() << endl;
		levels4Filters[tempj]=Convert(tmpffldMat);
		//cout << "eigen mat to level " << levels4Filters[tempj].rows() << " " << levels4Filters[tempj].cols() << endl;
		//levels4Filters[tempi]=bristowFilters[tempi];
	}

	cout << "transforming filters to cache " << endl;

	std::vector<FFLD::Patchwork::Filter> bristowFilterCache_(bristowFilters.size());
	for (int tempi=0;tempi<bristowFilters.size();tempi++){
		//std::vector<cv::Ptr<cv::FilterEngine> > tmpFilterEngine=bristowFilters[tempi];
		//cv::Ptr<cv::FilterEngine>  tmp2FilterEngine=tmpFilterEngine[0];
		//cout << " tmpFilter Enginer size " << tmp2FilterEngine.size() << " " << tmp2FilterEngine.rows << " " << tmp2FilterEngine.cols << endl;
		//Mat filterEngineI=tmpFilterEngine;
		FFLD::Patchwork::TransformFilter(levels4Filters[tempi],bristowFilterCache_[tempi]);
	}
	cout << "imported filters of size " << bristowFilters.size() << endl;

		cout << " Bristow filters transform to cache " << endl;

	vector<vector<FFLD::HOGPyramid::Matrix> > bristowConvolutions(bristowFilterCache_.size());
	patchworkBristow.convolve(bristowFilterCache_, bristowConvolutions);

	cout << " Done Convolution in FFLD, " << endl;

	cout << " FFLDs Bristow convolution size " << bristowConvolutions.size() << " " << bristowConvolutions[0].size() << endl;
	Mat C;
	vector2DMat pdfFFLD;
	//const int tmpnbFilters = 27, tmpnbPlanes = 3, tmpnbLevels = 10;
	/*pdfFFLD.resize(tmpnbLevels, vectorMat(tmpnbFilters * tmpnbPlanes));
		for (int i = 0; i < tmpnbFilters * tmpnbPlanes; ++i) {
		const int k = i / tmpnbPlanes; // Filter index
		const int l = i % tmpnbPlanes; // Plane index
		for (int j = 0; j < tmpnbLevels; ++j) {
			FFLD::HOGPyramid::Matrix ffldResponse = bristowConvolutions[k][j];
			FFLD::HOGPyramid::Matrix tempffldResponse;
			eigen2cv(ffldResponse,C);
			pdfFFLD[j][i]=C;
			//cout << "transfer i & j " << i << " " << j << endl;
			//cv2eigen(C,tempffldResponse);
			cout << " C dim convo ffld my modiefied " << i << " " << j << " " << C.rows << " " << C.cols << " "<< C.channels() << endl;
		}
	}*/
	pdfFFLD.resize(bristowConvolutions[0].size(), vectorMat(bristowConvolutions.size()));
	for (int i = 0; i < bristowConvolutions[0].size(); i++)
		for (int j=0; j < bristowConvolutions.size(); j++){
			FFLD::HOGPyramid::Matrix ffldResponse = bristowConvolutions[j][i];
			FFLD::HOGPyramid::Matrix tempffldResponse;
			eigen2cv(ffldResponse,C);
			pdfFFLD[i][j]=C;
			//cout << " C dim convo ffld my modiefied " << i << " " << j << " " << C.rows << " " << C.cols << " "<< C.channels() << endl;
		}
	// convolve the feature pyramid with the Part experts
	// to get probability density for each Part
	
	
	/////test code - Hussain
	//creat a test pyramid ffld of same dim as pyramid Bristow
	///creat empty bristow pyramid and copy the original pyramid to it from mat structure
	//levels_.resize(maxScale + 1);
	//levels_ = levels;


/*


	


	//responses.resize(M, vectorMat(N));

	//Patchwork::TransformFilter(const HOGPyramid::Level & filter, Filter & result);
	//FFLD::Mixture bristowMixture;
	//bristowFilterCache_ = bristowMixture.filterCacheObj();
	
	*/
	//split(feature.reshape(stride), featurev);

	cout << "FFLD Convolution Part Starting " << endl;
	
/*	int interval = 3;
	vector<FFLD::HOGPyramid::Matrix> scores;
	vector<FFLD::Mixture::Indices> argmaxes;
	vector<vector<vector<FFLD::Model::Positions> > > positions;
	const string file("../tmp.jpg");
	cout << file << endl;
	FFLD::JPEGImage image(file);
	FFLD::HOGPyramid pyramidFFLD(image, padding, padding, interval);
	//cout << "ffld pyramid size " << pyramidFFLD.size() << endl;
	FFLD::HOGPyramid::Matrix levelMat;
	
	
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
	vector2DMat pdfFFLD2;
	const int tmpnbFilters2 = 54, tmpnbPlanes2 = 4, tmpnbLevels2 = 13;
	pdfFFLD2.resize(tmpnbLevels2, vectorMat(tmpnbFilters2 * tmpnbPlanes2));
	cout << " convolutions size " << convolutions[0].size() << " " << convolutions.size() << endl;
		for (int i = 0; i < convolutions[0].size(); i++)
		for (int j=0; j < convolutions.size(); j++){
			FFLD::HOGPyramid::Matrix ffldResponse = convolutions[j][i];
			FFLD::HOGPyramid::Matrix tempffldResponse;
			eigen2cv(ffldResponse,C);
			pdfFFLD2[i][j]=C;
			cout << " C dim convo ffld real " << i << " " << j << " " << C.rows << " " << C.cols << " "<< C.channels() << endl;
		}*/
	cout << " done patch work convolution " << endl;
	//const int tmpnbFilters = 54, tmpnbPlanes = 12, tmpnbLevels = 41;// need the right dim here, after changing ffld matrix
	
	//just test code to transfer convolution matrix of ffld to vector2dMat of bristow
	/*
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
			//cout << "ffldResponse dim " << ffldResponse.rows() << " " << ffldResponse.cols() << " C dim " << C.rows << " " << C.cols << " " << C.type()  << " " << C.channels() <<" ffld::Eigen dim " << tempffldResponse.rows() << " " << tempffldResponse.cols() <<endl;
		}
	}
	*/
	double t = (double)getTickCount();
	vector2DMat pdf;
	cout << " Starting Convolution - in PBM-detect function " << endl;
	convolution_engine_->pdf(pyramid, pdf);
/*	cout << " convolution size " << pdf.size() << " " << pdf[0].size() << endl;
	for (int pdfInd=0;pdfInd<pdf.size();pdfInd++)
		for (int pdfInd2=0;pdfInd2<pdf[0].size();pdfInd2++){
		Mat response;
		response=pdf[pdfInd][pdfInd2];
		cout << "pdf dim " << response.rows << " " << response.cols << endl;
		response=pdfFFLD[pdfInd][pdfInd2];
		cout << "pdfFFLD dim " << response.rows << " " << response.cols << endl;
	}
	*/
	cout << " End Convolution - in PBM-detect function " << endl;
	printf("Convolution time: %f\n", ((double)getTickCount() - t)/getTickFrequency());

	// use dynamic programming to predict the best detection candidates from the part responses
	vector4DMat Ix, Iy, Ik;
	vector2DMat rootv, rooti;
	t = (double)getTickCount();
	//dp_.min(parts_, pdf, Ix, Iy, Ik, rootv, rooti);
	cout << " got here dp" << endl;
	dp_.min(parts_, pdfFFLD, Ix, Iy, Ik, rootv, rooti); // error is size of array
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
	pdbFilters=model.filters(); // hussain: filters to matvector format

	// initialize the tree of Parts
	parts_ = Parts(model.filters(), model.filtersi(), model.def(), model.defi(), model.bias(), model.biasi(),
			model.anchors(), model.biasid(), model.filterid(), model.defid(), model.parentid());

	// initialize the dynamic program
	dp_ = DynamicProgram<T>(model.thresh());

}



// declare all specializations of the template
template class PartsBasedDetector<float>;
template class PartsBasedDetector<double>;
