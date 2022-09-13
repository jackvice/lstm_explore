#include <explore/frontier_search.h>

#include <mutex>

#include <costmap_2d/cost_values.h>
#include <costmap_2d/costmap_2d.h>
#include <geometry_msgs/Point.h>

#include <explore/costmap_tools.h>

#include "jack.h"
#include <math.h> 
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <cmath>


namespace frontier_exploration
{
using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::NO_INFORMATION;
using costmap_2d::FREE_SPACE;

FrontierSearch::FrontierSearch(costmap_2d::Costmap2D* costmap,
                               double potential_scale, double gain_scale,
                               double min_frontier_size)//, double ssim)
  : costmap_(costmap)
  , potential_scale_(potential_scale)
  , gain_scale_(gain_scale)
  , min_frontier_size_(min_frontier_size)
    //, ssim_(ssim)
{
}
  /****  From header file
struct Frontier {
  std::uint32_t size;
  double min_distance;
  double cost;
  geometry_msgs::Point initial;
  geometry_msgs::Point centroid;
  geometry_msgs::Point middle;
  std::vector<geometry_msgs::Point> points;
  double ssim;
};
  ***/

std::vector<Frontier> FrontierSearch::searchFrom(geometry_msgs::Point position)  //take robot position
{
  //ROS_INFO("######################################################### FRONTIER SEARCHfrom");
  std::vector<Frontier> frontier_list;

  // Sanity check that robot is inside costmap bounds before searching
  unsigned int mx, my;
  if (!costmap_->worldToMap(position.x, position.y, mx, my)) {
    ROS_ERROR("Robot out of costmap bounds, cannot search for frontiers");
    return frontier_list;
  }

  // make sure map is consistent and locked for duration of search
  std::lock_guard<costmap_2d::Costmap2D::mutex_t> lock(*(costmap_->getMutex()));

  map_ = costmap_->getCharMap();
  size_x_ = costmap_->getSizeInCellsX();
  size_y_ = costmap_->getSizeInCellsY();

  // initialize flag arrays to keep track of visited and frontier cells
  std::vector<bool> frontier_flag(size_x_ * size_y_, false);
  std::vector<bool> visited_flag(size_x_ * size_y_, false);

  // initialize breadth first search
  std::queue<unsigned int> bfs; //standard c++ fifo queue

  // find closest clear cell to start search
  unsigned int clear, pos = costmap_->getIndex(mx, my);  // x and y of frontier
  if (nearestCell(clear, pos, FREE_SPACE, *costmap_)) {
    bfs.push(clear);
  } else {
    bfs.push(pos);
    ROS_WARN("Could not find nearby clear cell to start search");
  }
  visited_flag[bfs.front()] = true;
  const int hot_cold_size = 1024;
  static double hot_spots_x[hot_cold_size];
  static double hot_spots_y[hot_cold_size];
  static double cold_spots_x[hot_cold_size];
  static double cold_spots_y[hot_cold_size];
  static int hot_index = 0;
  static int cold_index = 0;

  static double min_ssim = 1.0;
  static double reference_x_old = 0.0;
  static double reference_y_old = 0.0;
  double reference_x_, reference_y_;
  using namespace Eigen;
  Quaternionf q;
  double angleToFront;
  double angleMoving;
  static bool is_first = true;
  if (is_first){
    for (int i = 0; i < hot_cold_size; i++){
      hot_spots_x[i] = 100;
      hot_spots_y[i] = 100;
      cold_spots_x[i] = 100;
      cold_spots_y[i] = 100;
    }
    is_first = false;
  }
  while (!bfs.empty()) {
    unsigned int idx = bfs.front();
    bfs.pop();

    // iterate over 4-connected neighbourhood
    for (unsigned nbr : nhood4(idx, *costmap_)) {
      // add to queue all free, unvisited cells, use descending search in case
      // initialized on non-free cell
      if (map_[nbr] <= map_[idx] && !visited_flag[nbr]) {  //is nbr a cell or a frontier

        visited_flag[nbr] = true;
        bfs.push(nbr);
        // check if cell is new frontier cell (unvisited, NO_INFORMATION, free
        // neighbour)
      } else if (isNewFrontierCell(nbr, frontier_flag)) {
        frontier_flag[nbr] = true;
        Frontier new_frontier = buildNewFrontier(nbr, pos, frontier_flag);
        if (new_frontier.size * costmap_->getResolution() >=
            min_frontier_size_) {
          //ROS_INFO("################################################# NEW FRONTIER");
          unsigned int rx_, ry_;
          
          costmap_->indexToCells(pos, rx_, ry_);
          costmap_->mapToWorld(rx_, ry_, reference_x_, reference_y_);
          double distanceNew = sqrt(pow((double(reference_x_) - double(new_frontier.middle.x)), 2.0) +
                                    pow((double(reference_y_) - double(new_frontier.middle.y)), 2.0));
          double distanceOld = sqrt(pow((double(reference_x_old) - double(new_frontier.middle.x)), 2.0) +
                                    pow((double(reference_y_old) - double(new_frontier.middle.y)), 2.0));
          double distanceDiff = distanceOld - distanceNew;
          double ssim_exp = 3;
          double ssim_mult = 4;
          double ssim_thresh = 0.84;
          double ssim_upper = 0.915; //0.88;
          if (distanceDiff > 0.5 && global_ssim < 0.86)
            {
              new_frontier.ssim = (1 / global_ssim) * 10;
              ROS_INFO("######################### directly front distance diff %.3f, global_ssim %.3f ", distanceDiff, global_ssim);
            }
          if ((abs(distanceDiff) > 0.08 && distanceNew < 1.5) || (distanceNew < 0.25))
            {
              if (global_ssim < ssim_thresh)
                {
                  new_frontier.ssim = (1 / global_ssim) * 10;//10; //pow( ( (1 - global_ssim) * ssim_mult), ssim_exp);
                  ROS_INFO("##### NEW SSIM Frontier robot at ref_x %.3f, ref_y %.3f,   frontier at x: %.3f, y: %.3f ,  SSIM_reward: %.3f,   ssim: %.3f",
                           reference_x_, reference_y_, new_frontier.middle.x, new_frontier.middle.y, new_frontier.ssim, global_ssim);

                  hot_spots_x[hot_index] = new_frontier.middle.x;
                  hot_spots_y[hot_index] = new_frontier.middle.y;
                  hot_index = hot_index + 1;
                }
              /*******
              else if (global_ssim > 0.92)
                { // not looking at anomaly
                  new_frontier.ssim = global_ssim * -5; // pow( ( (1 - .75) * ssim_mult), ssim_exp);
                  ROS_INFO("##### Increase Cost Frontier robot at ref_x %.3f, ref_y %.3f,   frontier at x: %.3f, y: %.3f ,  SSIM_reward: %.3f,   ssim: %.3f",
                           reference_x_, reference_y_, new_frontier.middle.x, new_frontier.middle.y, new_frontier.ssim, global_ssim);
                  ROS_INFO("################################################# cold index: %d ", cold_index);
                  cold_spots_x[cold_index] = new_frontier.middle.x;
                  cold_spots_y[cold_index] = new_frontier.middle.y;
                  cold_index = cold_index + 1;
                }
              **********/
            }
          
          if (global_ssim < min_ssim){
            min_ssim = global_ssim;
            }
          angleToFront = atan2 ( (new_frontier.middle.y - reference_y_), (new_frontier.middle.x - reference_x_) ) + 3.1415926535897932384626433832;
          //angleToFront = atan2 ( (new_frontier.middle.y - reference_y_), (new_frontier.middle.x - reference_x_) );
          angleMoving = atan2 ( (reference_y_ - reference_y_old), (reference_x_ - reference_x_old) ) + 3.1415926535897932384626433832;
          //auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
          //ROS_INFO("FRONTIER at %.3f %.3f, robot at %.3f %.3f,  angle to front: %.3f, angle moving: %.3f",
          //         new_frontier.middle.x, new_frontier.middle.y, reference_x_, reference_y_, angleToFront, angleMoving);
          //ROS_INFO("New and old y: %.3f,  %.3f, new and old x:  %.3f, %.3f", reference_y_ , reference_y_old, reference_x_ , reference_x_old );
          ROS_INFO("Frontier x %.3f y %.3f,  distance Old %.3f, distance New %.3f, diff %.3f, min_ssim %.3f, global_ssim %.3f ",
             new_frontier.middle.x, new_frontier.middle.y, distanceOld, distanceNew, distanceOld - distanceNew, min_ssim, global_ssim);
          
          frontier_list.push_back(new_frontier);  // this is where were push a new frontier.
        }
      }
    }
  }
  reference_x_old = reference_x_;
  reference_y_old = reference_y_;
  double distance = 0.0;
  // set costs of frontiers
  for (auto& frontier : frontier_list) {
    /******
    for (int i = 0; i < hot_index; i++){ // see if we close to a hot spot
      distance = sqrt( (frontier.middle.x - hot_spots_x[i])*(frontier.middle.x - hot_spots_x[i]) +
                                  (frontier.middle.y - hot_spots_y[i])*(frontier.middle.y - hot_spots_y[i]) );
      if (distance < 1.5){
        frontier.ssim =  global_ssim * 25;
        ROS_INFO("########################################### HOT update   x: %.3f y: %.3f, hot distance: %.3f ", hot_spots_x[i], hot_spots_y[i], distance);
      }  
    } 
    for (int i = 0; i < cold_index; i++){  // see if we close to a cold spot
      distance = sqrt( (frontier.middle.x - cold_spots_x[i])*(frontier.middle.x - cold_spots_x[i]) +
                                   (frontier.middle.y - cold_spots_y[i])*(frontier.middle.y - cold_spots_y[i]) );

      if (distance < .5){
        frontier.ssim =  -20;
        ROS_INFO("######################################### Cold update i: %d,  x: %.3f y: %.3f, cold distance: %.3f ",i, cold_spots_x[i], cold_spots_y[i], distance);
      }
    }
    ********/
    frontier.cost = frontierCost(frontier);
  }
  std::sort(
      frontier_list.begin(), frontier_list.end(),
      [](const Frontier& f1, const Frontier& f2) { return f1.cost < f2.cost; });
  for(Frontier i : frontier_list){
    //ROS_INFO("######################### Frontier List:  x: %.3f, y: %.3f, cost: %.3f, .ssim: %.3f ", i.middle.x, i.middle.y, i.cost, i.ssim );
  }
  return frontier_list;
}

Frontier FrontierSearch::buildNewFrontier(unsigned int initial_cell,
                                          unsigned int reference,
                                          std::vector<bool>& frontier_flag)
{
  // initialize frontier structure
  Frontier output;
  output.centroid.x = 0;
  output.centroid.y = 0;
  output.size = 1;
  output.min_distance = std::numeric_limits<double>::infinity();
  output.ssim =  0; //global_ssim;

  // record initial contact point for frontier
  unsigned int ix, iy;
  costmap_->indexToCells(initial_cell, ix, iy);
  costmap_->mapToWorld(ix, iy, output.initial.x, output.initial.y);
  //ROS_DEBUG("ix %.5d, iy %.5d, out.x %.5f, out.y %.5f", ix, iy, output.initial.x, output.initial.y);
  // push initial gridcell onto queue
  std::queue<unsigned int> bfs;
  bfs.push(initial_cell);

  // cache reference position in world coords
  unsigned int rx, ry;
  double reference_x, reference_y;
  costmap_->indexToCells(reference, rx, ry);
  costmap_->mapToWorld(rx, ry, reference_x, reference_y);
  //ROS_INFO("rx %.5d, ry %.5d, ref.x %.5f, ref.y %.5f", rx, ry, reference_x, reference_y); //refs are gazebo robot location
  while (!bfs.empty()) {
    unsigned int idx = bfs.front();
    bfs.pop();

    // try adding cells in 8-connected neighborhood to frontier
    for (unsigned int nbr : nhood8(idx, *costmap_)) {
      // check if neighbour is a potential frontier cell
      if (isNewFrontierCell(nbr, frontier_flag)) {
        // mark cell as frontier
        frontier_flag[nbr] = true;
        unsigned int mx, my;
        double wx, wy;
        costmap_->indexToCells(nbr, mx, my);
        costmap_->mapToWorld(mx, my, wx, wy);

        geometry_msgs::Point point;
        point.x = wx;
        point.y = wy;
        output.points.push_back(point);

        // update frontier size
        output.size++;

        // update centroid of frontier
        output.centroid.x += wx;
        output.centroid.y += wy;

        // determine frontier's distance from robot, going by closest gridcell
        // to robot
        double distance = sqrt(pow((double(reference_x) - double(wx)), 2.0) +
                               pow((double(reference_y) - double(wy)), 2.0));
        //ROS_INFO("Distance %.5f, ref.x %.5f, ref.y %.5f, wx  %.5f, wy  %.5f ",
        //distance, reference_x, reference_y, wx, wy);
        if (distance < output.min_distance) {
          output.min_distance = distance;
          output.middle.x = wx;
          output.middle.y = wy;
        }

        // add to queue for breadth first search
        bfs.push(nbr);
      }
    }
  }

  // average out frontier centroid
  output.centroid.x /= output.size;
  output.centroid.y /= output.size;
  //ROS_INFO("output.centroid.x %.5f, output.centroid.y %.5f,  output.min_distance %.5f,  output.cost %.5f",
  //         output.centroid.x, output.centroid.y,  output.min_distance, output.cost);
  return output;
}


bool FrontierSearch::isNewFrontierCell(unsigned int idx,
                                       const std::vector<bool>& frontier_flag)
{
  // check that cell is unknown and not already marked as frontier
  if (map_[idx] != NO_INFORMATION || frontier_flag[idx]) {
    return false;
  }

  // frontier cells should have at least one cell in 4-connected neighbourhood
  // that is free
  for (unsigned int nbr : nhood4(idx, *costmap_)) {
    if (map_[nbr] == FREE_SPACE) {
      return true;
    }
  }

  return false;
}


double FrontierSearch::frontierCost(const Frontier& frontier)
{
  //double ssim_part = pow( ( (1 - global_ssim) * 10), 2.0);
  //ROS_INFO("########################################################################## UPDATING FRONTIER COSTS");
  //ROS_INFO("SSIM %.5f, ssim_part %.5f,  costmap_->getResolution %.5f", frontier.ssim, ssim_part, costmap_->getResolution());
  return (potential_scale_ * frontier.min_distance *
          costmap_->getResolution()) -
    //(gain_scale_ * frontier.size * costmap_->getResolution());  // turn off ssim frontier
    (gain_scale_ * frontier.size * costmap_->getResolution()) - frontier.ssim; //pow( ( (1 - frontier.ssim) * 10), 2.0) ;
}
}
