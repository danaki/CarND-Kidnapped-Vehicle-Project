/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <iostream>
#include <ctime>
#include <iomanip>
#include <random>
#include <math.h>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 100;
	
	default_random_engine gen;
	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);
	
	for (int i = 0; i < num_particles; ++i) {
		Particle p = Particle();
		p.id = i;
		p.x = N_x(gen);
		p.y = N_y(gen);
		p.theta = N_theta(gen);
		p.weight = 1;
		particles.push_back(p);
		weights.push_back(0);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);
	
	for (int i = 0; i < num_particles; ++i) {
		Particle p = particles[i];
		p.x = p.x + velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + N_x(gen);
		p.y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + N_y(gen);
		p.theta = p.theta + yaw_rate * delta_t +  N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (int i = 0; i < observations.size(); ++i) {
		int min_j = -1;
		int min_d = -1;
		for (int j = 0; j < predicted.size(); ++j) {
			float d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (min_d < 0 || (d < min_d)) {
				min_d = d;
				min_j = j;
			}
		}
		
		observations[i] = predicted[min_j];		
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	
	for (int i = 0; i < num_particles; ++i) {
		vector<LandmarkObs> new_observations;
		
		// filter by sensor range
    	for (LandmarkObs &observation : observations) {
			if (dist(observation.x, observation.y, 0, 0) > sensor_range) {
				continue;
			}
			
			LandmarkObs obs;
			
			obs.x = observation.x * cos(particles[i].theta) - observation.y * sin(particles[i].theta) + particles[i].x;
			obs.y = observation.x * sin(particles[i].theta) + observation.y * cos(particles[i].theta) + particles[i].y;
			obs.id = -1;
			
			new_observations.push_back(obs);
		}
    
    	vector<LandmarkObs> new_landmarks;
    	for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
        	if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) > sensor_range) {
				continue;
			}
			
			LandmarkObs landmark;
			
			landmark.x = map_landmarks.landmark_list[j].x_f;
			landmark.y = map_landmarks.landmark_list[j].y_f;
			landmark.id = map_landmarks.landmark_list[j].id_i;
			
			new_landmarks.push_back(landmark);
		}
    
		dataAssociation(new_landmarks, new_observations);

		// likelihood
		double p = 1;
		for (LandmarkObs &observation : new_observations) {
			int k = observation.id;
			p *= exp(-(
					pow(observation.x - new_landmarks[k].x, 2)
						/ (2 * pow(std_landmark[0], 2))
							+
					pow(observation.y - new_landmarks[k].y, 2)
						/ (2 * pow(std_landmark[1], 2))))
				/ (2 * M_PI * std_landmark[0] * std_landmark[1]);      		
    	}

    	weights[i] = p;
    	particles[i].weight = p;
  	} 
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;
	default_random_engine gen;
	discrete_distribution<> d(weights.begin(), weights.end());
	
	for (int i = 0; i < num_particles; ++i) {
		new_particles.push_back(particles[d(gen)]);
	}
	
	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
