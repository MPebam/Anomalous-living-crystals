#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
//#include <omp.h>
#include <string>
#include <sstream>

class Simulator{

private:

    struct Constants {
        //constants
        static constexpr double PI = 3.14159265;
        static constexpr int T = 298; //K
        static constexpr double visc = 0.001; //N s m^-2
        static constexpr double kB = 1.38e-23; //J K-1
        static constexpr double delta_T = 1e-4; //s
        static constexpr double scaling = 0.25;
        static constexpr double beta = 0;
        static constexpr double gamma = 1;
        static constexpr double delta = 0;


        //number of particles
        static int num_particles(double density, double radius, double box_prop) {
            return static_cast<int>(density/((PI*radius*radius)/(pow(box_prop*radius,2))));
        }
        //cutoff calculations
        static double cutoff(double radius) {
            return 10*radius;
        }
        static double tor_cutoff(double radius) {
            return 8*radius;
        }
        static double cutoff_passive(double radius) {
            return 2*radius;
        }
        static double offset(double radius) {
            return 2*radius;
        }
        //force multiplier
        static double force_mult(double radius) {
            return 0.5/(6*PI*radius*visc);
        }
        static double torque_mult(double radius) {
            return (4*radius*radius)*delta_T/(8*PI*pow(radius,3)*visc);
        }
        //diffusion
        static double sqr_trans_diffusion(double radius) {
            return sqrt(2*delta_T*((kB*T)/(6*PI*radius*visc)));
        }
        static double sqr_rot_diffusion(double radius) {
            return sqrt(2*delta_T*((kB*T)/(8*PI*pow(radius,3)*visc)));
        }
        //sigma
        static double sigma(double radius) {
            return (2*radius)/(pow(2,1.0/6.0));
        }
        //well depth
        static double LJ_depth(double lj_dep) {
            return lj_dep*kB*T;
        }
        static double tor_depth(double tor_dep) {
            return tor_dep*kB*T;
        }
        //rebuild_counter
        static int rebuild_counter(double radius, double act_vel) {
            return static_cast<int>((0.5*radius)/(act_vel*delta_T));
        }
        //calculating power
        static double power(double value, int pows) {
            double res = 1.0;
            while(pows > 0){
                res *= value;
                --pows;
            }
            return res;
        }
    };

    //sim variables
    const int num_active_particles_;
    const int num_passive_particles_;
    const double radius_;
    const double act_part_vel_;
    const double sqr_trans_diffusion_;
    const double sqr_rot_diffusion_;
    const double size_box_;
    const double inv_size_box_;
    const double sigma_;
    const double sigma_6_;
    const double sigma_12_;
    const double lj_depth_;
    const double tor_depth_;
    const double cutoff_;
    const double tor_cutoff_;
    const double cutoff_passive_;
    const double offset_;
    const int buffer_size_;
    const int counter_record_pos_;
    int global_step_count_;
    int global_event_count_ = 0;
    const int construct_neigh_list_counter_;
    const double force_multiplier_;
    const double torque_multiplier_;
    std::random_device rd_trans_;
    std::mt19937 generator_trans_;
    std::normal_distribution<double> distribution_trans_;
    std::random_device rd_angle_;
    std::mt19937 generator_angle_;
    std::normal_distribution<double> distribution_angle_;
    std::random_device rd_lev_1_;
    std::mt19937 generator_lev_1_;
    std::normal_distribution<double> distribution_normal_lev_;
    std::random_device rd_lev_2_;
    std::mt19937 generator_lev_2_;
    std::uniform_real_distribution<double> distribution_uniform_lev_;

    //arrays
    std::vector<double> positions_x_ = std::vector<double>((num_passive_particles_+num_active_particles_));
    std::vector<double> positions_y_ = std::vector<double>((num_passive_particles_+num_active_particles_));
    std::vector<double> forces_x_ = std::vector<double>((num_passive_particles_+num_active_particles_));
    std::vector<double> forces_y_ = std::vector<double>((num_passive_particles_+num_active_particles_));
    std::vector<double> torques_ = std::vector<double>(num_active_particles_);
    std::vector<double> orientations_ = std::vector<double>(num_active_particles_);
    std::vector<double> run_lengths_ = std::vector<double>(num_active_particles_);
    std::vector<double> alphas_ = std::vector<double>(num_active_particles_);
    std::vector<int> part_tumbled_ = std::vector<int>(num_active_particles_);
    std::vector<int> neighbours_lj_;
    std::vector<int> neighbours_torque_;
    std::vector<int> place_holder_lj_ = std::vector<int>((num_passive_particles_+num_active_particles_));
    std::vector<int> place_holder_torque_ = std::vector<int>((num_active_particles_));

    std::vector<double> record_x_ = std::vector<double>((buffer_size_*num_active_particles_));
    std::vector<double> record_y_ = std::vector<double>((buffer_size_*num_active_particles_));
    std::vector<double> record_alpha_ = std::vector<double>((buffer_size_*num_active_particles_));
    std::vector<double> record_orient_ = std::vector<double>((buffer_size_*num_active_particles_));

    //private-helper constructor
    Simulator(double radius, int num_act_parts, int num_pass_parts, double act_vel, double sqr_trans, double sqr_rot, double size_box, double sig, double lj, double tor, double cut, double t_cut, double pass_cut, double off, int global_step_count, int construct_neigh_list_counter, double force_multiplier, double sigma_12, double sigma_6, double torque_multiplier, int counter_record_pos, int buffer_size)
    :   radius_(radius),
        num_active_particles_(num_act_parts),
        num_passive_particles_(num_pass_parts),
        act_part_vel_(act_vel),
        sqr_trans_diffusion_(sqr_trans),
        sqr_rot_diffusion_(sqr_rot),
        size_box_(size_box),
        sigma_(sig),
        lj_depth_(lj),
        tor_depth_(tor),
        cutoff_(cut),
        tor_cutoff_(t_cut),
        cutoff_passive_(pass_cut),
        offset_(off),
        inv_size_box_(1/size_box),
        global_step_count_(global_step_count),
        construct_neigh_list_counter_(construct_neigh_list_counter),
        force_multiplier_(force_multiplier),
        generator_trans_(rd_trans_()),
        distribution_trans_(0.0,1.0),
        generator_angle_(rd_angle_()),
        distribution_angle_(0.0,1.0),
        sigma_12_(sigma_12),
        sigma_6_(sigma_6),
        torque_multiplier_(torque_multiplier),
        generator_lev_1_(rd_lev_1_()),
        distribution_normal_lev_(0.0,1.0),
        generator_lev_2_(rd_lev_2_()),
        distribution_uniform_lev_(0.0,1.0),
        counter_record_pos_(counter_record_pos),
        buffer_size_(buffer_size){}

    //private functions
    //relaxed positions of the particles to ensure there are no overlaps
    bool relax_positions_rand(std::mt19937 (&generator), std::uniform_real_distribution<double> dis) {
        bool existing_overlaps = false;
        int total_overlaps = 0;

        double dx = 0.0;
        double dy = 0.0;
        double new_x = 0.0;
        double new_y = 0.0;

        for (int part_1 = 0; part_1 < (num_active_particles_+num_passive_particles_-1); part_1++) {
            double pos_1_x = positions_x_[part_1];
            double pos_1_y = positions_y_[part_1];
            for (int part_2 = part_1+1; part_2 < (num_active_particles_+num_passive_particles_); part_2++) {
                double pos_2_x = positions_x_[part_2];
                double pos_2_y = positions_y_[part_2];

                dx = pos_2_x-pos_1_x;
                dy = pos_2_y-pos_1_y;
                //particles overlap
                if ((dx*dx+dy*dy) < (4*radius_*radius_)) {
                    existing_overlaps = true;
                    total_overlaps++;
                    positions_x_[part_2] = dis(generator);
                    positions_y_[part_2] = dis(generator);
                }
            }
        }
        std::cout << "Current overlaps: " << total_overlaps << " out of " << (num_active_particles_+num_passive_particles_) << std::endl;
        return !existing_overlaps;
    }
    bool relax_positions_iterative(int max_iterations = 2000) {
        bool existing_overlaps;
        int total_overlaps = 0.0;

        double dx = 0.0;
        double dy = 0.0;
        double new_x = 0.0;
        double new_y = 0.0;
        double min_dist = (2*radius_);

        for (int iter = 0; iter < max_iterations; iter ++) {
            existing_overlaps = false;
            for (int part_1 = 0; part_1 < (num_active_particles_+num_passive_particles_-1); part_1++) {
                for (int part_2 = part_1+1; part_2 < (num_active_particles_+num_passive_particles_); part_2++) {
                    dx = positions_x_[part_2]-positions_x_[part_1];
                    dy = positions_y_[part_2]-positions_y_[part_1];
                    double dist = sqrt(dx*dx+dy*dy);
                    //particles overlap
                    if (dist < min_dist) {
                        existing_overlaps = true;
                        //displacement based
                        positions_x_[part_1] += dx/2;
                        positions_y_[part_1] += dy/2;
                        positions_x_[part_2] -= dx/2;
                        positions_y_[part_2] -= dy/2;
                        //adjusting for boundaries
                        positions_x_[part_1] = boundary_condition(positions_x_[part_1]);
                        positions_y_[part_1] = boundary_condition(positions_y_[part_1]);
                        positions_x_[part_2] = boundary_condition(positions_x_[part_2]);
                        positions_y_[part_2] = boundary_condition(positions_y_[part_2]);
                    }
                }
            }
        }
        return !existing_overlaps;
    }
    //constructs the neighbour list
    void build_neighbour_lists() {
        int count_position_lj_list = 0;
        int count_position_torque_list = 0;
        neighbours_lj_.clear();
        neighbours_torque_.clear();
        double cut_off_lj = 0.0;
        double cut_off_torque = ((tor_cutoff_+offset_)*(tor_cutoff_+offset_));
        double dx = 0.0;
        double dy = 0.0;
        //LJ neighbour list
        for (int part_1 = 0; part_1 < (num_active_particles_+num_passive_particles_-1); part_1 ++) {
            place_holder_lj_[part_1] = count_position_lj_list;
            if (part_1 < num_passive_particles_) {
                cut_off_lj = ((cutoff_passive_+offset_)*(cutoff_passive_+offset_));
            } else {
                cut_off_lj = ((cutoff_+offset_)*(cutoff_+offset_));
            }
            for (int part_2 = part_1+1; part_2 < (num_active_particles_+num_passive_particles_); part_2 ++) {
                dx = image_distance(positions_x_[part_2]-positions_x_[part_1]);
                dy = image_distance(positions_y_[part_2]-positions_y_[part_1]);
                //lj interaction happens between all particles
                if ((dx*dx+dy*dy) < cut_off_lj) {
                    //for lj interaction
                    count_position_lj_list++;
                    neighbours_lj_.push_back(part_2);
                }
            }
        }
        //torque neighbour list
        for (int part_1 = num_passive_particles_; part_1 < (num_passive_particles_+num_active_particles_); part_1++) {
            place_holder_torque_[part_1-num_passive_particles_] = count_position_torque_list;
            for (int part_2 = 0; part_2 < num_passive_particles_; part_2++) {
                dx = image_distance(positions_x_[part_2]-positions_x_[part_1]);
                dy = image_distance(positions_y_[part_2]-positions_y_[part_1]);
                //lj interaction happens between all particles
                if ((dx*dx+dy*dy) < cut_off_torque) {
                    //for lj interaction
                    count_position_torque_list++;
                    neighbours_torque_.push_back(part_2);
                }
            }
        }
    }
    //computes the forces
    void calculate_forces() {
        std::fill(forces_x_.begin(), forces_x_.end(),0.0);
        std::fill(forces_y_.begin(), forces_y_.end(),0.0);
        //loop through all the particles
        //#pragma omp parallel for schedule(dynamic)
        for (int part_1 = 0; part_1 < (num_passive_particles_+num_active_particles_-1); part_1++) {
            double pos_1_x = positions_x_[part_1];
            double pos_1_y = positions_y_[part_1];

            double relevant_cutoff = 0.0;

            if (part_1 < num_passive_particles_) {
                relevant_cutoff = cutoff_passive_*cutoff_passive_;
            } else {
                relevant_cutoff = cutoff_*cutoff_;
            }

            for (int neighbour = place_holder_lj_[part_1]; neighbour < place_holder_lj_[part_1+1]; neighbour++) {
                double dx = pos_1_x-positions_x_[neighbours_lj_[neighbour]];
                double dy = pos_1_y-positions_y_[neighbours_lj_[neighbour]];

                dx = image_distance(dx);
                dy = image_distance(dy);

                double sq_distance = dx*dx+dy*dy;
                if (sq_distance < relevant_cutoff) {
                    double distance = sqrt(sq_distance);
                    double force_magnitude = lennard_jones(distance);
                    double dx_project = dx/distance;
                    double dy_project = dy/distance;
                    //#pragma omp atomic
                    forces_x_[part_1] += force_magnitude*dx_project;
                    //#pragma omp atomic
                    forces_y_[part_1] += force_magnitude*dy_project;
                    //#pragma omp atomic
                    forces_x_[neighbours_lj_[neighbour]] -= force_magnitude*dx_project;
                    //#pragma omp atomic
                    forces_y_[neighbours_lj_[neighbour]] -= force_magnitude*dy_project;
                }
            }
        }
    }
    void calculate_torques() {
        std::fill(torques_.begin(), torques_.end(),0.0);

        for (int part_1 = num_passive_particles_; part_1 < num_active_particles_+num_passive_particles_; part_1++) {
            double pos_1_x = positions_x_[part_1];
            double pos_1_y = positions_y_[part_1];
            double unit_x = cos(orientations_[part_1-num_passive_particles_]);
            double unit_y = sin(orientations_[part_1-num_passive_particles_]);

            double relevant_cutoff = tor_cutoff_*tor_cutoff_;

            int end_index = 0;
            if (part_1 < num_active_particles_+num_passive_particles_-1) {
                end_index = place_holder_torque_[part_1-num_passive_particles_+1];
            } else {
                end_index = neighbours_torque_.size();
            }

            for (int neighbour = place_holder_torque_[part_1-num_passive_particles_]; neighbour < end_index; neighbour++) {

                double dx = pos_1_x-positions_x_[neighbours_torque_[neighbour]];
                double dy = pos_1_y-positions_y_[neighbours_torque_[neighbour]];

                dx = image_distance(dx);
                dy = image_distance(dy);

                double sq_distance = dx*dx+dy*dy;
                if (sq_distance < relevant_cutoff) {
                    double distance = sqrt(sq_distance);
                    torques_[part_1-num_passive_particles_] += torque(distance, dx, dy, unit_x, unit_y);
                }
            }
        }
    }
    //actual interactions calculations
    double lennard_jones(double distance) {
        double first = (2*sigma_12_) / Constants::power(distance, 13);
        double second = (sigma_6_) / Constants::power(distance,7);
        return 24*lj_depth_*(first-second);
    }
    double torque(double total_distance, double dist_x, double dist_y, double unit_x, double unit_y) {
        double exponent = -(exp(-(0.25/(2*radius_))*total_distance)/(total_distance*total_distance))*((0.25/(2*radius_))+(1/total_distance));
        double x_pot = dist_x*exponent;
        double y_pot = dist_y*exponent;
        return (unit_x*y_pot)-(unit_y*x_pot);
    }
    //integrates positions in timestep
    void integrate_positions() {
        //calculate forces
        calculate_forces();
        //calculate torques
        calculate_torques();
        //adjust positions
        for (int part = 0; part < num_active_particles_+num_passive_particles_; part ++) {
            double new_x = positions_x_[part] + forces_x_[part]*Constants::delta_T*force_multiplier_ + sqr_trans_diffusion_*distribution_trans_(generator_trans_);
            double new_y = positions_y_[part] + forces_y_[part]*Constants::delta_T*force_multiplier_ + sqr_trans_diffusion_*distribution_trans_(generator_trans_);
            if (part >= num_passive_particles_) {
                //checking if tumbling event should happen

                run_lengths_[part-num_passive_particles_]--;
                if (run_lengths_[part-num_passive_particles_] < 0) {

                    //run_lengths_[part-num_passive_particles_] += Constants::scaling*generate_sample_run(alphas_[part-num_passive_particles_])/Constants::delta_T;
                    part_tumbled_[part-num_passive_particles_] = 1;
                    orientations_[part-num_passive_particles_] += (Constants::PI/6)*distribution_angle_(generator_angle_);
                    //add to tumble event
                    global_event_count_++;
                }

                orientations_[part-num_passive_particles_] += torques_[part-num_passive_particles_]*torque_multiplier_*tor_depth_;

                new_x += act_part_vel_*Constants::delta_T*cos(orientations_[part-num_passive_particles_]);
                new_y += act_part_vel_*Constants::delta_T*sin(orientations_[part-num_passive_particles_]);
            }

            new_x = boundary_condition(new_x);
            new_y = boundary_condition(new_y);

            positions_x_[part] = new_x;
            positions_y_[part] = new_y;
        }
    }
    //boundary condition
    double image_distance(double d_1d) {
        return d_1d -= size_box_*round(d_1d*inv_size_box_);
    }
    double boundary_condition(double pos_1d) {
        if (pos_1d < 0) pos_1d += size_box_;
        else if (pos_1d >= size_box_) pos_1d -= size_box_;
        return pos_1d;
    }
    double generate_sample_run(double alph) {
        double run_len = 0.0;

        if (alph == 2.0) {
            run_len = sqrt(2)*distribution_normal_lev_(generator_lev_2_);
        }
        else if (alph == 1.0 && Constants::beta == 0.0) {
            run_len = tan(Constants::PI/2 * (2 * distribution_uniform_lev_(generator_lev_1_) - 1));
        }
        else if (alph == 0.5 && abs(Constants::beta) == 1.0) {
            run_len = Constants::beta / (pow(distribution_normal_lev_(generator_lev_2_), 2));
        }
        else if (Constants::beta == 0.0) {
            float V = (Constants::PI/2) * (2 * distribution_uniform_lev_(generator_lev_1_) - 1);
            float W = -log(distribution_uniform_lev_(generator_lev_2_));
            run_len = (sin(alph*V)/pow(cos(V),1/alph))*pow((cos(V*(1-alph))/W),(1-alph)/alph);
        }
        else if (alph != 1) {
            float V = (Constants::PI/2) * (2 * distribution_uniform_lev_(generator_lev_1_) - 1);
            float W = -log(distribution_uniform_lev_(generator_lev_2_));
            float con = Constants::beta * tan(Constants::PI*alph/2);
            float B = atan(con);
            float S = pow((1+(con*con)),1/(2*alph));
            run_len = (S*sin((alph*V)+B)/pow(cos(V),1/alph))*pow((cos((V*(1-alph))-B)/W),(1-alph)/alph);
        }
        else {
            float V = (Constants::PI/2) * (2 * distribution_uniform_lev_(generator_lev_2_) - 1);
            float W = -log(distribution_uniform_lev_(generator_lev_1_));
            float sclshiftV = (Constants::PI/2) + Constants::beta*V;
            run_len = (2/Constants::PI) * ((sclshiftV*tan(V))-Constants::beta*(log((Constants::PI*W*cos(V)/2)/sclshiftV)));
        }
        if (alph != 1.0) {
            run_len = (run_len*Constants::gamma) + Constants::delta;
        }
        else {
            run_len = (run_len*Constants::gamma) + ((2/Constants::PI)*Constants::beta*Constants::gamma*log(Constants::gamma)) + Constants::delta;
        }

        return std::abs(run_len);
    }

    void record_positions() {
        int index = (global_step_count_/counter_record_pos_) % buffer_size_;

        for (int part = 0; part < num_active_particles_; part ++) {
            record_x_[(buffer_size_*part)+index] = positions_x_[part+num_passive_particles_];
            record_y_[(buffer_size_*part)+index] = positions_y_[part+num_passive_particles_];
            record_orient_[(buffer_size_*part)+index] = orientations_[part];
            record_alpha_[(buffer_size_*part)+index] = alphas_[part];
        }
    }
    void set_run_lengths() {
        for (int part = 0; part < num_active_particles_; part ++) {
            if (part_tumbled_[part]) {
                run_lengths_[part-num_passive_particles_] += Constants::scaling*generate_sample_run(alphas_[part-num_passive_particles_])/Constants::delta_T;
            }
            part_tumbled_[part] = 0;
        }
    }

public:

    //constructor
    Simulator(double radius, double dens_active, double dens_passive, double lj_dep, double tor_dep, double act_vel, int global_step_count, double box_prop, int counter_record_pos, int buffer_size)
    :   Simulator(radius,
                Constants::num_particles(dens_active, radius, box_prop),
                Constants::num_particles(dens_passive, radius, box_prop),
                act_vel,
                Constants::sqr_trans_diffusion(radius),
                Constants::sqr_rot_diffusion(radius),
                box_prop*radius,
                Constants::sigma(radius),
                Constants::LJ_depth(lj_dep),
                Constants::tor_depth(tor_dep),
                Constants::cutoff(radius),
                Constants::tor_cutoff(radius),
                Constants::cutoff_passive(radius),
                Constants::offset(radius),
                global_step_count,
                Constants::rebuild_counter(radius, act_vel),
                Constants::force_mult(radius),
                Constants::power(Constants::sigma(radius), 12),
                Constants::power(Constants::sigma(radius), 6),
                Constants::torque_mult(radius),
                counter_record_pos,
                buffer_size) {}

    //resets the simlation state
    void reset() {



    }
    void load_data_from_file(std::string load_file) {

        std::ifstream checkpoint_file(load_file);
        std::string line = "";
        std::getline(checkpoint_file, line);

        std::vector<double> data;
        int loc = 0;
        //loading data with the positions, orientations and walk lengths of all particles
        while (line.length() > 1) {
            loc = line.find(" ");
            data.push_back(std::stod(line.substr(0, loc)));
            line.erase(0, loc+1);
        }
        //dumping the data into the relevant arrays
        //passive positions
        for (int part = 0; part < num_passive_particles_; part++) {
            positions_x_[part] = data[part*2];
            positions_y_[part] = data[part*2+1];
        }
        //active particle positions, orientations, run lengths and alphas
        for (int part = 0; part < num_active_particles_; part++) {
            positions_x_[part+num_passive_particles_] = data[part*5+(num_passive_particles_*2)];
            positions_y_[part+num_passive_particles_] = data[1+(part*5)+(num_passive_particles_*2)];
            orientations_[part] = data[2+(part*5)+(num_passive_particles_*2)];
            run_lengths_[part] = data[3+(part*5)+(num_passive_particles_*2)];
            alphas_[part] = data[4+(part*5)+(num_passive_particles_*2)];
        }
        checkpoint_file.close();
    }
    //initialize data
    void initialize_positions() {
        std::random_device rand_place_device;
        std::mt19937 generator_position(rand_place_device());
        std::uniform_real_distribution<double> dis(0, size_box_);
        //positions for passive particles
        for (int part = 0; part < (num_active_particles_+num_passive_particles_); part ++) {
            double x_pos = dis(generator_position);
            double y_pos = dis(generator_position);
            positions_x_[part] = x_pos;
            positions_y_[part] = y_pos;
        }
        //random placement results in particle overlaps, so positions must be relaxed
        bool positions_relaxed = false;
        int temp_counter = 0;
        while (!positions_relaxed) {
            positions_relaxed = relax_positions_iterative();
            if (!positions_relaxed) {
                positions_relaxed = relax_positions_rand(generator_position, dis);
            }
            if (temp_counter % 1 == 0) {
                std::cout << "Relaxing positions: " << temp_counter << std::endl;
            }
            temp_counter++;
        }
    }
    void initialize_orientations() {
        std::random_device rand_dev_orient;
        std::mt19937 generator_orient(rand_dev_orient());
        std::uniform_real_distribution<double> dis_orient(0, 1);
        for (int part = 0; part < num_active_particles_; part++) {
            orientations_[part] = (2*Constants::PI*dis_orient(generator_orient))-Constants::PI;
        }
    }
    void initialize_alphas(double same_value = -1.0) {
        if (same_value == -1) {
            std::random_device rand_dev_alpha;
            std::mt19937 generator_alpha(rand_dev_alpha());
            std::uniform_real_distribution<double> dis_alpha(1, 2);
            for (int part = 0; part < num_active_particles_; part++) {
                alphas_[part] = dis_alpha(generator_alpha);
            }
        } else {
            for (int part = 0; part < num_active_particles_; part++) {
                alphas_[part] = same_value;
            }
        }
    }

    void print_data() {
        for (int part = 0; part < num_active_particles_+num_passive_particles_; part++) {
            std::cout << positions_x_[part] << ", " << positions_y_[part];
            if (part >= num_passive_particles_) {
                std::cout << ", " << alphas_[part-num_passive_particles_] << ", " << orientations_[part-num_passive_particles_];
            }
            std::cout << std::endl;
        }
    }
    void save_positions(std::string filename, bool is_checkpoint = false) {

        std::ofstream save_file;
        if (is_checkpoint) {
            save_file.open(filename, std::ofstream::trunc);
        } else {
            save_file.open(filename, std::ios_base::app);
        }
        for (int part = 0; part < (num_active_particles_+num_passive_particles_); part ++) {
            save_file << positions_x_[part] << " " << positions_y_[part];
            if (part >= num_passive_particles_) {
                save_file << " " << orientations_[part-num_passive_particles_] << " " << run_lengths_[part-num_passive_particles_] << " " << alphas_[part-num_passive_particles_];
            }
            if (!is_checkpoint) {
                save_file << ";";
            } else {
                save_file << " ";
            }
        }
        save_file.close();
    }

    void initialize_walk_lengths() {
        for (int part = num_passive_particles_; part < num_active_particles_+num_passive_particles_; part++) {
            run_lengths_[part-num_passive_particles_] = Constants::scaling*generate_sample_run(alphas_[part-num_passive_particles_])/Constants::delta_T;
        }
    }

    //run time driven simulation
    void run_time_simulation(int num_steps){
        for (int step = 0; step < num_steps; step++) {

            if (global_step_count_ % construct_neigh_list_counter_ == 0) {
                build_neighbour_lists();
            }
            integrate_positions();
            global_step_count_++;
        }
    }
    //run event driven simulation
    void run_event_simulation(int num_events = 1) {

        //draw new run_length for the tumbled particles
        set_run_lengths();

        while (global_event_count_ < num_events) {

            if (global_step_count_ % construct_neigh_list_counter_ == 0) {
                build_neighbour_lists();
            }
            integrate_positions();
            global_step_count_++;
            if (global_step_count_ % counter_record_pos_ == 0) {
                record_positions();
            }

        }

        //once the number of events surpasses num_events, reset the counter
        global_event_count_ = 0;
    }

    //getters
    const std::vector<double>& get_recorded_positions_x() const {
        return record_x_;
    }
    const std::vector<double>& get_recorded_positions_y() const {
        return record_y_;
    }
    const std::vector<double>& get_recorded_orientations() const {
        return record_orient_;
    }
    const std::vector<double>& get_recorded_alphas() const {
        return record_alpha_;
    }
    const std::vector<int>& get_part_tumbled() const {
        return part_tumbled_;
    }
    const int get_step_count() const {
        return global_step_count_;
    }
    const int get_number_active_particles() const {
        return num_active_particles_;
    }
    //setters
    void set_alpha(int part, double new_alpha) {
        alphas_[part] = new_alpha;
    }

};

/*
int main() {

    //omp_set_num_threads(4);

    Simulator new_simulation(0.69e-6, 0.015, 0.1, 30, -5, 6e-6, 0, 200);
    std::cout << "Object created" << std::endl;

    new_simulation.initialize_positions();
    new_simulation.initialize_alphas(1.0);
    new_simulation.initialize_orientations();
    new_simulation.initialize_walk_lengths();

    //new_simulation.save_positions("test_event_driven.txt");
    new_simulation.run_event_simulation(10);
    std::cout << "Pause" << std::endl;
    new_simulation.run_event_simulation(50);


    for (int iters = 0; iters < 1000; iters ++) {
        std::cout << iters << std::endl;
        new_simulation.run_time_simulation(5000);
        new_simulation.save_positions("test_event_driven.txt");
    }


    return 0;
}*/

