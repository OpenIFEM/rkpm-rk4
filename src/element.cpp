#include "element.h"

template <int dim>
element<dim>::element(const std::vector<unsigned int>& indices)
    : nodes(indices) {}

template class element<2>;
template class element<3>;

// Assuming node ids increase in Y, X directions sequentially,
// so the element at the (Xmin, Ymin) corner would be (0, 4, 5, 1)
std::vector<element<2>> element_map_rectangle(unsigned int ex,
                                              unsigned int ey) {
  std::vector<element<2>> map;

  unsigned int ny = ey + 1;
  const unsigned int pattern[4] = {0, 1, ny + 1, ny};
  unsigned int inc = 0;

  for (unsigned int i = 0; i < ex; i++) {
    for (unsigned int j = 0; j < ey; j++) {
      map.push_back(element<2>(
          std::vector<unsigned int>{pattern[0] + inc, pattern[3] + inc,
                                    pattern[2] + inc, pattern[1] + inc}));
      inc++;
    }
    inc = (i + 1) * ny;
  }

  return map;
}

std::vector<std::vector<element<2>>> boundary_element_rectangle(
    unsigned int ex, unsigned int ey) {
  std::vector<element<2>> lower, right, upper, left;

  unsigned int ny = ey + 1;
  const unsigned int pattern[4] = {0, 1, ny + 1, ny};
  unsigned int inc = 0;

  for (unsigned int i = 0; i < ex; i++) {
    for (unsigned int j = 0; j < ey; j++) {
      element<2> e(std::vector<unsigned int>{pattern[0] + inc, pattern[3] + inc,
                                             pattern[2] + inc,
                                             pattern[1] + inc});
      if (i == 0) {
        left.push_back(e);
      }
      if (i == ex - 1) {
        right.push_back(e);
      }
      if (j == 0) {
        lower.push_back(e);
      }
      if (j == ey - 1) {
        upper.push_back(e);
      }
      inc++;
    }
    inc = (i + 1) * ny;
  }
  return {lower, right, upper, left};
}

// Assuming node ids increase in X, Y, Z directions sequentially,
// so the element at the (Xmin, Ymin, Zmin) corner would be
// (0, 1, nx+1, nx, nx*ny, 1+nx*ny, nx+1+nx*ny, nx+nx*ny)
std::vector<element<3>> element_map_rectangle(unsigned int ex, unsigned int ey,
                                              unsigned int ez) {
  std::vector<element<3>> map;
  unsigned int nx = ex + 1, ny = ey + 1, nz = ez + 1;
  std::vector<unsigned int> pattern{
      0, 1, nx + 1, nx, nx * ny, 1 + nx * ny, nx + 1 + nx * ny, nx + nx * ny};
  for (unsigned int k = 0; k < ez; k++) {
    for (unsigned int j = 0; j < ey; j++) {
      for (unsigned int i = 0; i < ex; i++) {
        unsigned int offset = i + j * nx + k * nx * ny;
        auto v = pattern;
        for (unsigned int n = 0; n < v.size(); ++n) v[n] += offset;
        map.push_back(element<3>(v));
      }
    }
  }
  return map;
}

std::vector<std::vector<element<3>>> boundary_element_rectangle(
    unsigned int ex, unsigned int ey, unsigned int ez) {
  std::vector<std::vector<element<3>>> faces(6);
  unsigned int nx = ex + 1, ny = ey + 1, nz = ez + 1;
  std::vector<unsigned int> pattern{
      0, 1, nx + 1, nx, nx * ny, 1 + nx * ny, nx + 1 + nx * ny, nx + nx * ny};
  for (unsigned int k = 0; k < ez; k++) {
    for (unsigned int j = 0; j < ey; j++) {
      for (unsigned int i = 0; i < ex; i++) {
        unsigned int offset = i + j * nx + k * nx * ny;
        auto v = pattern;
        for (unsigned int n = 0; n < v.size(); ++n) v[n] += offset;
        element<3> e(v);
        if (k == 0)  // -Z
          faces[0].push_back(e);
        if (k == ez - 1)  // +Z
          faces[1].push_back(e);
        if (j == 0)  // -Y
          faces[2].push_back(e);
        if (i == ex - 1)  // +X
          faces[3].push_back(e);
        if (j == ey - 1)  // +Y
          faces[4].push_back(e);
        if (i == 0)  // -X
          faces[5].push_back(e);
      }
    }
  }
  return faces;
}
