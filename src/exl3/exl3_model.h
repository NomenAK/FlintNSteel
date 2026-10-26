// Flint EXL3 safetensors loader: views over mapped files, no weight copies.
#pragma once
#include "fns/model.h"
#include <map>
#include <memory>
namespace fns {
struct Exl3Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const void *data = nullptr;
    size_t bytes = 0;
};
class Exl3Model {
  public:
    explicit Exl3Model(const std::string &directory);
    ~Exl3Model();
    Exl3Model(const Exl3Model &) = delete;
    Exl3Model &operator=(const Exl3Model &) = delete;
    const Exl3Tensor &tensor(const std::string &name) const;
    Hparams hp;
    std::vector<int32_t> eog_ids;
    std::vector<bool> swa;

  private:
    struct Mapping;
    std::vector<std::unique_ptr<Mapping>> files_;
    std::map<std::string, Exl3Tensor> tensors_;
};
} // namespace fns
