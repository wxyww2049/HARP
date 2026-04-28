#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>

#include <vector>
#include <algorithm>
#include <cstdint>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace {

struct BroadcastMeta {
  std::vector<int64_t> shape;
  std::vector<int64_t> a_strides;
  std::vector<int64_t> b_strides;
  int64_t ndim = 0;
};

static int64_t numel_from_shape(const std::vector<int64_t>& shape) {
  if (shape.empty()) return 1;
  int64_t n = 1;
  for (auto s : shape) n *= s;
  return n;
}

static BroadcastMeta make_broadcast_meta(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.device().is_cpu(), "sve_add: a must be CPU");
  TORCH_CHECK(b.device().is_cpu(), "sve_add: b must be CPU");
  TORCH_CHECK(a.scalar_type() == at::kFloat, "sve_add: a must be float32");
  TORCH_CHECK(b.scalar_type() == at::kFloat, "sve_add: b must be float32");

  const int64_t ndim = std::max<int64_t>(a.dim(), b.dim());
  BroadcastMeta meta;
  meta.ndim = ndim;
  meta.shape.resize(ndim);
  meta.a_strides.resize(ndim);
  meta.b_strides.resize(ndim);

  auto a_sizes = a.sizes();
  auto b_sizes = b.sizes();
  auto a_strides = a.strides();
  auto b_strides = b.strides();

  for (int64_t i = 0; i < ndim; ++i) {
    const int64_t a_i = i - (ndim - a.dim());
    const int64_t b_i = i - (ndim - b.dim());

    const int64_t a_size = (a_i >= 0) ? a_sizes[a_i] : 1;
    const int64_t b_size = (b_i >= 0) ? b_sizes[b_i] : 1;

    TORCH_CHECK(
        a_size == b_size || a_size == 1 || b_size == 1,
        "sve_add: broadcast failed at dim ", i,
        ", a_size=", a_size, ", b_size=", b_size);

    const int64_t out_size = std::max<int64_t>(a_size, b_size);
    meta.shape[i] = out_size;

    if (a_i < 0 || (a_size == 1 && out_size != 1)) {
      meta.a_strides[i] = 0;
    } else {
      meta.a_strides[i] = a_strides[a_i];
    }

    if (b_i < 0 || (b_size == 1 && out_size != 1)) {
      meta.b_strides[i] = 0;
    } else {
      meta.b_strides[i] = b_strides[b_i];
    }
  }

  return meta;
}

static void add_strided_lastdim_scalar(
    const float* pa,
    const float* pb,
    float* po,
    int64_t n,
    int64_t sa,
    int64_t sb) {
  for (int64_t i = 0; i < n; ++i) {
    po[i] = pa[i * sa] + pb[i * sb];
  }
}

static void add_scalar_strided_lastdim_scalar(
    const float* pa,
    float s,
    float* po,
    int64_t n,
    int64_t sa) {
  for (int64_t i = 0; i < n; ++i) {
    po[i] = pa[i * sa] + s;
  }
}

#if defined(__ARM_FEATURE_SVE)
static void add_contig_sve(const float* pa, const float* pb, float* po, int64_t n) {
  int64_t i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)n);
    svfloat32_t va = svld1(pg, pa + i);
    svfloat32_t vb = svld1(pg, pb + i);
    svfloat32_t vc = svadd_f32_x(pg, va, vb);
    svst1(pg, po + i, vc);
    i += svcntw();
  }
}

static void add_a_broadcast_sve(float a_scalar, const float* pb, float* po, int64_t n) {
  int64_t i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)n);
    svfloat32_t va = svdup_n_f32(a_scalar);
    svfloat32_t vb = svld1(pg, pb + i);
    svfloat32_t vc = svadd_f32_x(pg, va, vb);
    svst1(pg, po + i, vc);
    i += svcntw();
  }
}

static void add_b_broadcast_sve(const float* pa, float b_scalar, float* po, int64_t n) {
  int64_t i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)n);
    svfloat32_t va = svld1(pg, pa + i);
    svfloat32_t vb = svdup_n_f32(b_scalar);
    svfloat32_t vc = svadd_f32_x(pg, va, vb);
    svst1(pg, po + i, vc);
    i += svcntw();
  }
}

static void add_scalar_contig_sve(const float* pa, float s, float* po, int64_t n) {
  int64_t i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)n);
    svfloat32_t va = svld1(pg, pa + i);
    svfloat32_t vs = svdup_n_f32(s);
    svfloat32_t vc = svadd_f32_x(pg, va, vs);
    svst1(pg, po + i, vc);
    i += svcntw();
  }
}
#endif

static at::Tensor sve_add_impl(const at::Tensor& a, const at::Tensor& b) {
  auto meta = make_broadcast_meta(a, b);
  auto out = at::empty(meta.shape, a.options().dtype(at::kFloat));

  const float* a_ptr = a.data_ptr<float>();
  const float* b_ptr = b.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();

  if (meta.ndim == 0 || meta.shape.empty()) {
    out_ptr[0] = a_ptr[0] + b_ptr[0];
    return out;
  }

  std::vector<int64_t> out_strides(meta.ndim, 1);
  for (int64_t i = meta.ndim - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * meta.shape[i + 1];
  }

  const int64_t last = meta.ndim - 1;
  const int64_t inner = meta.shape[last];
  const int64_t sa_last = meta.a_strides[last];
  const int64_t sb_last = meta.b_strides[last];

  if (meta.ndim == 1) {
#if defined(__ARM_FEATURE_SVE)
    if (sa_last == 1 && sb_last == 1) {
      add_contig_sve(a_ptr, b_ptr, out_ptr, inner);
    } else if (sa_last == 0 && sb_last == 1) {
      add_a_broadcast_sve(a_ptr[0], b_ptr, out_ptr, inner);
    } else if (sa_last == 1 && sb_last == 0) {
      add_b_broadcast_sve(a_ptr, b_ptr[0], out_ptr, inner);
    } else {
      add_strided_lastdim_scalar(a_ptr, b_ptr, out_ptr, inner, sa_last, sb_last);
    }
#else
    add_strided_lastdim_scalar(a_ptr, b_ptr, out_ptr, inner, sa_last, sb_last);
#endif
    return out;
  }

  std::vector<int64_t> outer_shape(meta.shape.begin(), meta.shape.end() - 1);
  const int64_t outer_total = numel_from_shape(outer_shape);

  at::parallel_for(0, outer_total, 0, [&](int64_t begin, int64_t end) {
    for (int64_t linear = begin; linear < end; ++linear) {
      int64_t tmp = linear;
      int64_t a_off = 0;
      int64_t b_off = 0;
      int64_t o_off = 0;

      for (int64_t d = meta.ndim - 2; d >= 0; --d) {
        const int64_t idx = tmp % meta.shape[d];
        tmp /= meta.shape[d];
        a_off += idx * meta.a_strides[d];
        b_off += idx * meta.b_strides[d];
        o_off += idx * out_strides[d];
      }

      const float* pa = a_ptr + a_off;
      const float* pb = b_ptr + b_off;
      float* po = out_ptr + o_off;

#if defined(__ARM_FEATURE_SVE)
      if (sa_last == 1 && sb_last == 1) {
        add_contig_sve(pa, pb, po, inner);
      } else if (sa_last == 0 && sb_last == 1) {
        add_a_broadcast_sve(pa[0], pb, po, inner);
      } else if (sa_last == 1 && sb_last == 0) {
        add_b_broadcast_sve(pa, pb[0], po, inner);
      } else {
        add_strided_lastdim_scalar(pa, pb, po, inner, sa_last, sb_last);
      }
#else
      add_strided_lastdim_scalar(pa, pb, po, inner, sa_last, sb_last);
#endif
    }
  });

  return out;
}

static at::Tensor sve_add_scalar_impl(const at::Tensor& a, const at::Scalar& s) {
  TORCH_CHECK(a.device().is_cpu(), "sve_add_scalar: a must be CPU");
  TORCH_CHECK(a.scalar_type() == at::kFloat, "sve_add_scalar: a must be float32");

  const float scalar = s.toFloat();
  auto out = at::empty_like(a, a.options().dtype(at::kFloat));

  const float* a_ptr = a.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();

  if (a.dim() == 0) {
    out_ptr[0] = a_ptr[0] + scalar;
    return out;
  }

  const auto sizes = a.sizes();
  const auto strides = a.strides();
  const int64_t ndim = a.dim();

  const int64_t inner = sizes[ndim - 1];
  const int64_t sa_last = strides[ndim - 1];

  if (ndim == 1) {
#if defined(__ARM_FEATURE_SVE)
    if (sa_last == 1) {
      add_scalar_contig_sve(a_ptr, scalar, out_ptr, inner);
    } else {
      add_scalar_strided_lastdim_scalar(a_ptr, scalar, out_ptr, inner, sa_last);
    }
#else
    add_scalar_strided_lastdim_scalar(a_ptr, scalar, out_ptr, inner, sa_last);
#endif
    return out;
  }

  std::vector<int64_t> out_strides(ndim, 1);
  for (int64_t i = ndim - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * sizes[i + 1];
  }

  std::vector<int64_t> outer_shape(sizes.begin(), sizes.end() - 1);
  const int64_t outer_total = numel_from_shape(outer_shape);

  at::parallel_for(0, outer_total, 0, [&](int64_t begin, int64_t end) {
    for (int64_t linear = begin; linear < end; ++linear) {
      int64_t tmp = linear;
      int64_t a_off = 0;
      int64_t o_off = 0;

      for (int64_t d = ndim - 2; d >= 0; --d) {
        const int64_t idx = tmp % sizes[d];
        tmp /= sizes[d];
        a_off += idx * strides[d];
        o_off += idx * out_strides[d];
      }

      const float* pa = a_ptr + a_off;
      float* po = out_ptr + o_off;

#if defined(__ARM_FEATURE_SVE)
      if (sa_last == 1) {
        add_scalar_contig_sve(pa, scalar, po, inner);
      } else {
        add_scalar_strided_lastdim_scalar(pa, scalar, po, inner, sa_last);
      }
#else
      add_scalar_strided_lastdim_scalar(pa, scalar, po, inner, sa_last);
#endif
    }
  });

  return out;
}

static at::Tensor& sve_add_out_impl(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  auto r = sve_add_impl(a, b);
  out.resize_(r.sizes());
  out.copy_(r);
  return out;
}

static at::Tensor& sve_add_scalar_out_impl(const at::Tensor& a, const at::Scalar& s, at::Tensor& out) {
  auto r = sve_add_scalar_impl(a, s);
  out.resize_(r.sizes());
  out.copy_(r);
  return out;
}

} // namespace

TORCH_LIBRARY(sve_add, m) {
  m.def("add(Tensor a, Tensor b) -> Tensor");
  m.def("add_out(Tensor a, Tensor b, Tensor(a!) out) -> Tensor(a!)");
  m.def("add_scalar(Tensor a, Scalar s) -> Tensor");
  m.def("add_scalar_out(Tensor a, Scalar s, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(sve_add, CPU, m) {
  m.impl("add", TORCH_FN(sve_add_impl));
  m.impl("add_out", TORCH_FN(sve_add_out_impl));
  m.impl("add_scalar", TORCH_FN(sve_add_scalar_impl));
  m.impl("add_scalar_out", TORCH_FN(sve_add_scalar_out_impl));
}