#ifndef ELEMENT_H_
#define ELEMENT_H_

#include <vector>

template <int dim>
class element {
 public:
  std::vector<unsigned int> nodes;
  element(const std::vector<unsigned int>& indices);
};

std::vector<element<2>> element_map_rectangle(unsigned int ex, unsigned int ey);
std::vector<std::vector<element<2>>> boundary_element_rectangle(
    unsigned int ex, unsigned int ey);

std::vector<element<3>> element_map_rectangle(unsigned int ex, unsigned int ey,
                                              unsigned int ez);
std::vector<std::vector<element<3>>> boundary_element_rectangle(
    unsigned int ex, unsigned int ey, unsigned int ez);
#endif /* ELEMENT_H_ */

extern template class element<2>;
extern template class element<3>;
