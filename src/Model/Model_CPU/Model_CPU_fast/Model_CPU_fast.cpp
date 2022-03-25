// #define GALAX_MODEL_CPU_FAST 1
#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>
// #include <typeinfo>
#include <iostream>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast ::Model_CPU_fast(const Initstate &initstate, Particles &particles)
    : Model_CPU(initstate, particles)
{
}

// ! choose version
#define VERSION 2

#if VERSION == 1
void forward(int n_particles, const Initstate &initstate, Particles &particles, std::vector<float> &velocitiesx, std::vector<float> &velocitiesy, std::vector<float> &velocitiesz, std::vector<float> &accelerationsx, std::vector<float> &accelerationsy, std::vector<float> &accelerationsz)
{

    // ! Use Parfor
    // omp_set_num_threads(4);
#pragma omp parallel for
    for (int i = 0; i < n_particles; i++)
    // for (int j = 0; j < n_particles; j++)
    {
        for (int j = 0; j < n_particles; j++)
        // for (int i = 0; i < n_particles; i++)
        {
            if (i != j)
            {
                const float diffx = particles.x[j] - particles.x[i];
                const float diffy = particles.y[j] - particles.y[i];
                const float diffz = particles.z[j] - particles.z[i];

                float dij = diffx * diffx + diffy * diffy + diffz * diffz;
                // G = 10
                if (dij < 1.0) // two bodies are too close
                {
                    dij = 10.0; // dij = G
                }
                else
                {
                    dij = std::sqrt(dij);
                    dij = 10.0 / (dij * dij * dij); // dij=G/d^3
                }

                accelerationsx[i] += diffx * dij * initstate.masses[j];
                accelerationsy[i] += diffy * dij * initstate.masses[j];
                accelerationsz[i] += diffz * dij * initstate.masses[j];
            }
        }
    }
    // ? more parallelisme ?
#pragma omp parallel for
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

#elif VERSION == 2
void forward(int n_particles, const Initstate &initstate, Particles &particles, std::vector<float> &velocitiesx, std::vector<float> &velocitiesy, std::vector<float> &velocitiesz, std::vector<float> &accelerationsx, std::vector<float> &accelerationsy, std::vector<float> &accelerationsz)
{

    // ! Use Xsimd

    std::size_t inc = b_type::size;
#pragma omp parallel for
    for (int i = 0; i < n_particles; i += inc)
    {
        // load registers body i
        b_type rposx_i = b_type::load_unaligned(&particles.x[i]); // ? const
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
        b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
        b_type rvely_i = b_type::load_unaligned(&velocitiesy[i]);
        b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);

        for (int j = 0; j < n_particles; j += 1)
        {
            if (i != j)
            {
                // std::cout << "i" << i << std::endl;
                // std::cout << "j" << j << std::endl;

                // load registers body i
                const b_type rposx_j = b_type::load_unaligned(&particles.x[j]); // ? const
                const b_type rposy_j = b_type::load_unaligned(&particles.y[j]);
                const b_type rposz_j = b_type::load_unaligned(&particles.z[j]);
                const b_type raccx_j = b_type::load_unaligned(&accelerationsx[j]);
                const b_type raccy_j = b_type::load_unaligned(&accelerationsy[j]);
                const b_type raccz_j = b_type::load_unaligned(&accelerationsz[j]);
                const b_type rvelx_j = b_type::load_unaligned(&velocitiesx[j]);
                const b_type rvely_j = b_type::load_unaligned(&velocitiesy[j]);
                const b_type rvelz_j = b_type::load_unaligned(&velocitiesz[j]);

                // std::cout << "rposx_i" << rposx_i << std::endl;
                // std::cout << "rposy_i" << rposy_i << std::endl;
                // std::cout << "rposz_i" << rposz_i << std::endl;

                // std::cout << "rposx_j" << rposx_j << std::endl;
                // std::cout << "rposy_j" << rposy_j << std::endl;
                // std::cout << "rposz_j" << rposz_j << std::endl;

                b_type diffx = rposx_j - rposx_i;
                b_type diffy = rposy_j - rposy_i;
                b_type diffz = rposz_j - rposz_i;

                // std::cout << "diffx" << diffx << std::endl;
                // std::cout << "diffy" << diffy << std::endl;
                // std::cout << "diffz" << diffz << std::endl;

                b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;

                // b_type a = 1.0;
                b_type b = 10.0;
                // dij = xs::rsqrt(dij); // rsqrt=1/sqrt
                // dij = xs::select(xs::gt(a, dij), b, 10.0 * dij * dij * dij);

                // if (xs::gt(a,dij))
                // {
                //     dij = 10.0;
                // }
                // else
                // {
                //     dij = xs::sqrt(dij);
                //     dij = 10.0 / (dij * dij * dij);
                // }

                b_type c = xs::rsqrt(dij);
                /// dij = 1.0;
                dij = xs::fmin(b, 10.0 * c * c * c);
                // std::cout << "dij" << dij << std::endl;

                raccx_i = raccx_i + diffx * dij * initstate.masses[j];
                raccy_i = raccy_i + diffy * dij * initstate.masses[j];
                raccz_i = raccz_i + diffz * dij * initstate.masses[j];
            }
        }

        raccx_i.store_unaligned(&accelerationsx[i]);
        raccy_i.store_unaligned(&accelerationsy[i]);
        raccz_i.store_unaligned(&accelerationsz[i]);

        //         rvelx_i += raccx_i * 2.0f;
        //         rvely_i += raccy_i * 2.0f;
        //         rvelz_i += raccz_i * 2.0f;

        //         rvelx_i.store_unaligned(&velocitiesx[i]);
        //         rvely_i.store_unaligned(&velocitiesy[i]);
        //         rvelz_i.store_unaligned(&velocitiesz[i]);

        //         rposx_i += rvelx_i * 0.1f;
        //         rposy_i += rvely_i * 0.1f;
        //         rposz_i += rvelz_i * 0.1f;

        //         rposx_i.store_unaligned(&particles.x[i]);
        //         rposy_i.store_unaligned(&particles.y[i]);
        //         rposz_i.store_unaligned(&particles.z[i]);
    }
    // #pragma omp parallel for
    for (int i = 0; i < n_particles; i++)
    {
        velocitiesx[i] += accelerationsx[i] * 2.0f;
        // if (i==0) std::cout << "accelerationsx_" << i << " = " << accelerationsx[i] << std::endl;
        velocitiesy[i] += accelerationsy[i] * 2.0f;
        velocitiesz[i] += accelerationsz[i] * 2.0f;
        particles.x[i] += velocitiesx[i] * 0.1f;
        particles.y[i] += velocitiesy[i] * 0.1f;
        particles.z[i] += velocitiesz[i] * 0.1f;
    }
}
#elif VERSION == 3
void forward(int n_particles, const Initstate &initstate, Particles &particles, std::vector<float> &velocitiesx, std::vector<float> &velocitiesy, std::vector<float> &velocitiesz, std::vector<float> &accelerationsx, std::vector<float> &accelerationsy, std::vector<float> &accelerationsz)
{

}
#else
    std::cerr << "Not implemented!" << std::endl;
#endif

void Model_CPU_fast ::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

    forward(n_particles, initstate, particles, velocitiesx, velocitiesy, velocitiesz, accelerationsx, accelerationsy, accelerationsz);
}
#endif // GALAX_MODEL_CPU_FAST
