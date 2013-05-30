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
 *  File:    main.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 27, 2012
 */

#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/filesystem.hpp>
#include "PartsBasedDetector.hpp"
#include "Candidate.hpp"
#include "FileStorageModel.hpp"
#ifdef WITH_MATLABIO
	#include "MatlabIOModel.hpp"
#endif
#include "Visualize.hpp"
#include "types.hpp"
#include "nms.hpp"
#include "Rect3.hpp"
#include "DistanceTransform.hpp"

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
#ifndef _WIN32
#include <sys/time.h>


timeval Start, Stop;

inline void start()
{
	gettimeofday(&Start, 0);
}

inline int stop()
{
	gettimeofday(&Stop, 0);
	
	timeval duration;
	timersub(&Stop, &Start, &duration);
	
	return duration.tv_sec * 1000 + (duration.tv_usec + 500) / 1000;
}
#endif


using namespace cv;
//using namespace FFLD;
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

template<typename _Tp>
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic>& dst )
{
    dst.resize(src.rows, src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
             dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
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
        Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
}


int main(int argc, char** argv) {

	int padding = 12;
	int interval = 10;

	// check arguments
	if (argc != 3 && argc != 4) {
		printf("Usage: PartsBasedDetector model_file image_file [depth_file]\n");
		exit(-1);
	}

	vector<FFLD::HOGPyramid::Matrix> scores;
	vector<FFLD::Mixture::Indices> argmaxes;
	vector<vector<vector<FFLD::Model::Positions> > > positions;
	const string file("../tmp.jpg");
	cout << file << endl;
	FFLD::JPEGImage image(file);
	FFLD::HOGPyramid pyramidFFLD(image, padding, padding, interval);
	if (!FFLD::Patchwork::Init((pyramidFFLD.levels()[0].rows() - padding + 15) & ~15,
							(pyramidFFLD.levels()[0].cols() - padding + 15) & ~15)) {
		cerr << "\nCould not initialize the Patchwork class" << endl;
		return -1;
	}
		
	cout << "Initialized FFTW in " << stop() << " ms" << endl;
	FFLD::Mixture mixture;
	string modelffld("../model_car_final-1.2.4.txt");
	cout << modelffld << endl;
	ifstream in(modelffld.c_str(), ios::binary);
	
	if (!in.is_open()) {
		//showUsage();
		cerr << "\nInvalid model file " << modelffld << endl;
		return -1;
	}
	in >> mixture;

	//mixture.convolve(pyramidFFLD, scores, argmaxes, &positions);//full ffld run
	cout << "Convolution :p " << endl;
	const int nbModels = mixture.models().size();
	const int nbLevels = pyramidFFLD.levels().size();
	// Convolve with all the models
	vector<vector<FFLD::HOGPyramid::Matrix> > tmp(nbModels);
	//convolve(pyramid, tmp, positions);//going inside
		// Transform the filters if needed
#pragma omp critical
	//if (mixture.filterCache_.empty())
		mixture.cacheFilters();
	
	// Create a patchwork
	const FFLD::Patchwork patchwork(pyramidFFLD);
		// Convolve the patchwork with the filters
	std::vector<FFLD::Patchwork::Filter> filterCache_;
	 filterCache_ = mixture.filterCacheObj();
	vector<vector<FFLD::HOGPyramid::Matrix> > convolutions(filterCache_.size());
	patchwork.convolve(filterCache_, convolutions);///convolve patch with filters, 
	 const int tmpnbFilters = 54, tmpnbPlanes = 12, tmpnbLevels = 41;
	 //nb filters is number of filters (Read from model.txt)
	 // nblevels is number of HOGPyramid levels
	// nbplanes comes from converting pyramid to patchwork, rectangle logic ?
	 Mat C;
	for (int i = 0; i < tmpnbFilters * tmpnbPlanes; ++i) {
		const int k = i / tmpnbPlanes; // Filter index
		const int l = i % tmpnbPlanes; // Plane index
		for (int j = 0; j < tmpnbLevels; ++j) {
			FFLD::HOGPyramid::Matrix ffldResponse = convolutions[k][j];
			FFLD::HOGPyramid::Matrix tempffldResponse;
			eigen2cv(ffldResponse,C);

			cout << "ffldResponse dim " << ffldResponse.rows() << " " << ffldResponse.cols() << " C dim " << C.rows << " " << C.cols << endl;
		}
	}
	
	//cout << " convolution size " << convolutions.size() << "rows " << convolutions[1].rows() << endl;
	// convert the convolusions Eigen matrix to cvMat obj

	cout << " Mixture model size " << nbModels << endl;



	// determine the type of model to read
	cout << " Now starting Bristow " << endl;
	boost::scoped_ptr<Model> model;
	string ext = boost::filesystem::path(argv[1]).extension().string();
	if (ext.compare(".xml") == 0 || ext.compare(".yaml") == 0) {
		model.reset(new FileStorageModel);
	}
#ifdef WITH_MATLABIO
	else if (ext.compare(".mat") == 0) {
		model.reset(new MatlabIOModel);
	}
#endif
	else {
		printf("Unsupported model format: %s\n", ext.c_str());
		exit(-2);
	}
	bool ok = model->deserialize(argv[1]);
	if (!ok) {
		printf("Error deserializing file\n");
		exit(-3);
	}

	// create the PartsBasedDetector and distribute the model parameters
	PartsBasedDetector<float> pbd;
	pbd.distributeModel(*model);

	
	// load the image from file
	Mat_<float> depth;
	Mat im = imread(argv[2]);
        if (im.empty()) {
            printf("Image not found or invalid image format\n");
            exit(-4);
        }
	if (argc == 4) {
		depth = imread(argv[3], CV_LOAD_IMAGE_ANYDEPTH);
		// convert the depth image from mm to m
		depth = depth / 1000.0f;
	}
	
	// detect potential candidates in the image
	double t = (double)getTickCount();
	vector<Candidate> candidates;
	pbd.detect(im, depth, candidates);
	cout << " image dim " << im.rows << " " << im.cols << endl;
	printf("Detection time: %f\n", ((double)getTickCount() - t)/getTickFrequency());
	printf("Number of candidates: %ld\n", candidates.size());

	// display the best candidates
	Visualize visualize(model->name());
	SearchSpacePruning<float> ssp;
        Mat canvas;
	if (candidates.size() > 0) {
	    Candidate::sort(candidates);
	    Candidate::nonMaximaSuppression(im, candidates, 0.2);
	    visualize.candidates(im, candidates, canvas, true);
            visualize.image(canvas);
	    waitKey();
	}
	return 0;
}
