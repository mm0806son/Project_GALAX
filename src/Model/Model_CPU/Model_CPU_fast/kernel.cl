__kernel void compute_accelerations(
    __global float* x,
    __global float* y,
    __global float* z,
    __global float* masses,
    __global float* accelerationsx,
    __global float* accelerationsy,
    __global float* accelerationsz,
    const unsigned int n_particles)
{
    int i = get_global_id(0);
    if (i < n_particles)
    {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        for (int j = 0; j < n_particles; j++)
        {
            if (i != j)
            {
                float dx = x[j] - x[i];
                float dy = y[j] - y[i];
                float dz = z[j] - z[i];
                float distance_sq = dx*dx + dy*dy + dz*dz + 1e-6f;
                float force = masses[j] / (distance_sq * sqrt(distance_sq));
                ax += dx * force;
                ay += dy * force;
                az += dz * force;
            }
        }
        accelerationsx[i] = ax;
        accelerationsy[i] = ay;
        accelerationsz[i] = az;
    }
}
