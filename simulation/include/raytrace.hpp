#pragma once
#include <vector>

#include "geodesic.hpp"
#include "metric.hpp"

// Backward ray tracing: one light ray per image pixel, fired from a camera and
// followed backward through the Kerr geometry until it ends somewhere we can
// colour. A ray that falls through the horizon is black (the shadow); a ray that
// crosses the equatorial accretion disk picks up its glow; a ray that escapes to
// infinity sees the dark sky. At a = 0 the Kerr metric is Schwarzschild, so the
// same code draws both.

namespace bhsim {

// Everything the renderer needs to know about the viewpoint and the disk.
struct Camera {
    double a      = 0.0;    // black-hole spin (0 = Schwarzschild)
    double incl   = 1.3963; // inclination of the view from the spin axis, radians
                            // (80 degrees: nearly edge-on, so the disk is a band)
    double r_cam  = 1000.0; // camera distance from the hole, in units of M
    double half_width = 12.0; // half the image width on the sky, in units of M
    int    res    = 200;    // output is res x res pixels
    double r_in   = 0.0;    // disk inner edge; <= 0 means the ISCO for this spin
                            // (6 M at a = 0, about 2.32 M at a = 0.9)
    double r_out  = 20.0;   // disk outer edge
};

// Initial photon state for the pixel whose offsets on the image plane are
// (alpha, beta), both in units of M. The projection is orthographic: every ray
// starts on the image plane and flies parallel to the view axis.
KerrState camera_ray(const Camera& cam, double alpha, double beta);

// Brightness seen along one pixel's ray: 0 for the horizon or the empty sky,
// positive where the ray meets the disk (brighter nearer the inner edge).
double trace_pixel(const Kerr& bh, const Camera& cam, double alpha, double beta);

// Render the whole frame, row-major, length res * res. Row 0 is the top of the
// image. Parallelised over rows with OpenMP.
std::vector<double> render_image(const Camera& cam);

}  // namespace bhsim
