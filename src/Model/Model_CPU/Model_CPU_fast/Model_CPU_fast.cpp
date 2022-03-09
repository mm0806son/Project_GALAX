
#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast ::Model_CPU_fast(const Initstate &initstate, Particles &particles)
    : Model_CPU(initstate, particles)
{
}

/*
void forward(int n_particles, const Initstate &initstate, Particles &particles, std::vector<float> &velocitiesx, std::vector<float> &velocitiesy, std::vector<float> &velocitiesz, std::vector<float> &accelerationsx, std::vector<float> &accelerationsy, std::vector<float> &accelerationsz)
    {

        // ! Use Parfor
        //omp_set_num_threads(4);
#pragma omp parallel for
        for (int i = 0; i < n_particles; i++)
        {
            for (int j = 0; j < n_particles; j++)
            {
                if (i != j)
                {
                    const float diffx = particles.x[j] - particles.x[i];
                    const float diffy = particles.y[j] - particles.y[i];
                    const float diffz = particles.z[j] - particles.z[i];

                    float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                    if (dij < 1.0)
                    {
                        dij = 10.0;
                    }
                    else
                    {
                        dij = std::sqrt(dij);
                        dij = 10.0 / (dij * dij * dij);
                    }

                    accelerationsx[i] += diffx * dij * initstate.masses[j];
                    accelerationsy[i] += diffy * dij * initstate.masses[j];
                    accelerationsz[i] += diffz * dij * initstate.masses[j];
                }
            }
        }

// ? logn
#pragma omp parallel for
        for (int i = 0; i < n_particles; i++)
        {
            for (int j = i+1; j < n_particles; j++)
            {
                
                const float diffx = particles.x[j] - particles.x[i];
                const float diffy = particles.y[j] - particles.y[i];
                const float diffz = particles.z[j] - particles.z[i];

                float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                if (dij < 1.0)
                {
                    dij = 10.0;
                }
                else
                {
                    dij = std::sqrt(dij);
                    dij = 10.0 / (dij * dij * dij);
                }

                // ? would be faster ?
                // ? dij = min(10.0,10.0 / (dij * dij * dij))
                float temp = dij * initstate.masses[j];
                accelerationsx[i] += diffx * temp;
                accelerationsy[i] += diffy * temp;
                accelerationsz[i] += diffz * temp;

                temp = dij * initstate.masses[i];
                accelerationsx[j] -= diffx * temp;
                accelerationsy[j] -= diffy * temp;
                accelerationsz[j] -= diffz * temp;
                
            }
        }

// ? more parallelisme ?
// ? #pragma omp parallel for
        for (int i = 0; i < n_particles; i++)
        {
            velocitiesx[i] += accelerationsx[i] * 2.0f;
            velocitiesy[i] += accelerationsy[i] * 2.0f;
            velocitiesz[i] += accelerationsz[i] * 2.0f;
            particles.x[i] += velocitiesx[i] * 0.1f;
            particles.y[i] += velocitiesy[i] * 0.1f;
            particles.z[i] += velocitiesz[i] * 0.1f;
        }

        // OMP + xsimd version
        // #pragma omp parallel for
        //     for (int i = 0; i < n_particles; i += b_type::size)
        //     {
        //         // load registers body i
        //         const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        //         const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        //         const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        //               b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        //               b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        //               b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        //         ...
        //     }

    }

*/

void forward(int n_particles, const Initstate &initstate, Particles &particles, std::vector<float> &velocitiesx, std::vector<float> &velocitiesy, std::vector<float> &velocitiesz, std::vector<float> &accelerationsx, std::vector<float> &accelerationsy, std::vector<float> &accelerationsz)
    {

        // ! Use Xsimd

     

        using b_type = xsimd::batch<double, xsimd::avx>;
        std::size_t inc = b_type::size;
        #pragma omp parallel for
            for (int i = 0; i < n_particles; i += b_type::size)
            {
                // load registers body i
                const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
                const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
                const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
                      b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
                      b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
                      b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);


                for (int j = 0; j < n_particles; j++)
                {
                    if (i != j)
                    {
                        // load registers body i
                        const b_type rposx_j = b_type::load_unaligned(&particles.x[j]);
                        const b_type rposy_j = b_type::load_unaligned(&particles.y[j]);
                        const b_type rposz_j = b_type::load_unaligned(&particles.z[j]);
                            b_type raccx_j = b_type::load_unaligned(&accelerationsx[j]);
                            b_type raccy_j = b_type::load_unaligned(&accelerationsy[j]);
                            b_type raccz_j = b_type::load_unaligned(&accelerationsz[j]);

                        b_type diffx = rposx_j - rposx_i;
                        b_type diffy = rposy_j - rposy_i;
                        b_type diffz = rposz_j - rposz_i;

                        b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;

                        if (dij < 1.0)
                        {
                            dij = 10.0;
                        }
                        else
                        {
                            dij = std::sqrt(dij);
                            dij = 10.0 / (dij * dij * dij);
                        }

                        b_type raccx_i = raccx_i + diffx * dij * initstate.masses[j];
                        b_type raccy_i = raccy_i + diffy * dij * initstate.masses[j];
                        b_type raccz_i = raccz_i + diffz * dij * initstate.masses[j];

                        raccx_i.store_unaligned(&accelerationsx[i]);
                        raccy_i.store_unaligned(&accelerationsy[i]);
                        raccz_i.store_unaligned(&accelerationsz[i]);
                    }
                    
                }

            for (int i = 0; i < n_particles; i++)
            {
                velocitiesx[i] += accelerationsx[i] * 2.0f;
                velocitiesy[i] += accelerationsy[i] * 2.0f;
                velocitiesz[i] += accelerationsz[i] * 2.0f;
                particles.x[i] += velocitiesx[i] * 0.1f;
                particles.y[i] += velocitiesy[i] * 0.1f;
                particles.z[i] += velocitiesz[i] * 0.1f;
            }

    }
    }


void Model_CPU_fast ::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

    forward(n_particles, initstate, particles, velocitiesx, velocitiesy, velocitiesz, accelerationsx, accelerationsy, accelerationsz);
   
}
#endif // GALAX_MODEL_CPU_FAST
