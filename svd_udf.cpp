#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1

#include <vector>
#include <complex>
#include <string>
#include <sstream>
#include <cmath>
#include <lapacke.h>       // Make sure this is found via -I/opt/homebrew/include, etc.
#include <cblas.h>         // If you also need CBLAS calls

#include <iostream>        // For optional debugging/logging

// Maximum bond dimension
static const int MAX_BOND = 5;

/**
 * A structure to hold partial aggregates (the data from each row).
 */
struct SvdAggCtx {
    // Arrays for i, j, k, l
    std::vector<int> ti;
    std::vector<int> tj;
    std::vector<int> tk;
    std::vector<int> tl;

    // Arrays for real and imaginary parts
    std::vector<double> valsRe;
    std::vector<double> valsIm;

    // Remember the dimension parameters from the last step call
    int leftDim  = 0;  // l
    int rightDim = 0;  // r
};

/**
 * Step function: called once per row of the query.
 */
static void svdStep(
    sqlite3_context *context, 
    int argc, 
    sqlite3_value **argv
){
    // We expect 8 arguments: i, j, k, l, re, im, leftDim (lf), rightDim (rt)
    if (argc < 8) {
        // Not enough arguments; do nothing or raise an error
        return;
    }
    // Get aggregator context
    auto *p = (SvdAggCtx*)sqlite3_aggregate_context(context, sizeof(SvdAggCtx));
    if (!p) {
        // Out of memory
        std::cerr << "ERROR: aggregator context is null" << std::endl;
        return;
    }

    int i   = sqlite3_value_int(argv[0]);
    int j   = sqlite3_value_int(argv[1]);
    int k   = sqlite3_value_int(argv[2]);
    int l   = sqlite3_value_int(argv[3]);
    double re = sqlite3_value_double(argv[4]);
    double im = sqlite3_value_double(argv[5]);
    int lf  = sqlite3_value_int(argv[6]);
    int rt  = sqlite3_value_int(argv[7]);

    // Store the data in the aggregator context
    p->ti.push_back(i);
    p->tj.push_back(j);
    p->tk.push_back(k);
    p->tl.push_back(l);

    p->valsRe.push_back(re);
    p->valsIm.push_back(im);

    // Overwrite dims each row (assuming they're the same for all rows)
    p->leftDim  = lf;
    p->rightDim = rt;
}

/**
 * Finalize function: called after all step() calls finish.
 * Performs the SVD, truncates to MAX_BOND, and returns a JSON string.
 */
static void svdFinalize(sqlite3_context *context){
    // Get aggregator context
    auto *p = (SvdAggCtx*)sqlite3_aggregate_context(context, 0);
    if (!p) {
        // No data aggregated
        sqlite3_result_null(context);
        return;
    }

    // If no entries, return NULL
    if (p->valsRe.empty()) {
        sqlite3_result_null(context);
        return;
    }

    int l = p->leftDim;  // "l" from your aggregator logic
    int r = p->rightDim; // "r" from your aggregator logic

    // Reconstruct a (2*l) x (2*r) matrix in row-major order, complex data.
    int nRows = 2*l;
    int nCols = 2*r;

    // We'll store it as an interleaved complex array: re, im, re, im,...
    // So total size = nRows * nCols * 2 doubles.
    std::vector<double> A(2 * nRows * nCols, 0.0);

    // Fill the matrix:
    //   row = 2*ti + tj
    //   col = r*tk + tl
    //   value = re + i*im
    for (size_t idx = 0; idx < p->valsRe.size(); ++idx) {
        int row = 2*p->ti[idx] + p->tj[idx];
        int col = r*p->tk[idx] + p->tl[idx];

        if (row < nRows && col < nCols) {
            double re = p->valsRe[idx];
            double im = p->valsIm[idx];

            // Interleaved: each matrix element M[row,col] is 2*(row*nCols + col).
            int base = 2 * (row*nCols + col);
            A[base + 0] = re;  // real part
            A[base + 1] = im;  // imag part
        }
    }

    // Prepare for SVD: We'll use LAPACKE_zgesvd (the simpler variant),
    // which does NOT require manual workspace arrays.
    lapack_int m = nRows;
    lapack_int n = nCols;
    lapack_int lda = nCols;  // row-major => leading dimension = # of columns

    // min(m, n)
    lapack_int minDim = (m < n) ? m : n;

    // U will be (m x minDim), V^T will be (n x minDim)
    // in complex double format
    std::vector<double> U(2 * m * minDim, 0.0);
    std::vector<double> V(2 * n * minDim, 0.0);
    std::vector<double> S(minDim, 0.0);           // singular values
    // "superb" is used by zgesvd internally; size = minDim - 1
    std::vector<double> superb( (minDim > 1) ? (minDim - 1) : 1, 0.0 );

    // Reinterpret casts so LAPACKE sees complex double pointers
    lapack_complex_double* A_ptr = 
        reinterpret_cast<lapack_complex_double*>(A.data());
    lapack_complex_double* U_ptr = 
        reinterpret_cast<lapack_complex_double*>(U.data());
    lapack_complex_double* V_ptr = 
        reinterpret_cast<lapack_complex_double*>(V.data());

    // We call zgesvd with jobu='S' (small U) and jobvt='S' (small V^T).
    lapack_int info = LAPACKE_zgesvd(
        LAPACK_ROW_MAJOR,  // matrix_layout
        'S',               // jobu: compute min(m,n) columns of U
        'S',               // jobvt: compute min(m,n) rows of V^T
        m,                 // number of rows
        n,                 // number of cols
        A_ptr,             // input matrix
        lda,               // leading dimension
        S.data(),          // output singular values
        U_ptr,             // left singular vectors (U)
        m,                 // ldu = #rows in U = m
        V_ptr,             // right singular vectors (V^T)
        n,                 // ldvt = #cols in V^T = n
        superb.data()      // workspace for zgesvd
    );

    if (info != 0) {
        // SVD failed
        sqlite3_result_null(context);
        return;
    }

    // U is (m x minDim) in row-major; V is (n x minDim) in row-major (but it’s actually V^T).
    // Now we can limit dimension to MAX_BOND if minDim > MAX_BOND.
    int finalDim = (minDim > MAX_BOND) ? MAX_BOND : minDim;

    // Truncate S, U, V
    S.resize(finalDim);
    // Keep only the first finalDim columns in U
    // We have m x minDim, but we only want m x finalDim
    // => effectively chop the extra columns
    U.resize(2 * m * finalDim);

    // Similarly, chop V to n x finalDim
    V.resize(2 * n * finalDim);

    // In the Python aggregator, you do:
    //  U -> shape(l, 2, finalDim)
    //  multiply S into V => shape(finalDim, 2, r)
    // Let’s replicate that.

    // 1) Multiply each column c of V by S[c].
    for (int c = 0; c < finalDim; ++c) {
        double sigma = S[c];
        // For row-major: V[row, col=c], row in [0..(n-1)]
        for (int row = 0; row < n; ++row) {
            int base = 2 * (row*finalDim + c); 
            V[base + 0] *= sigma; // real
            V[base + 1] *= sigma; // imag
        }
    }

    // Build JSON
    //  "U_re":[ [ [u(l=0, ph=0,  c=0..), ...], [u(l=0, ph=1, c=0..), ...] ], ... ],
    //  "U_im": [...],
    //  "Vh_re":[ finalDim x 2 x r ],
    //  "Vh_im": [...]
    //  "sh":[ [l, finalDim], [finalDim, r] ]
    // In row-major, U is (m=2*l x finalDim).
    //   row = [0..(2*l)-1], col=[0..(finalDim-1)]
    //   row0 = row/2, row1 = row%2

    std::ostringstream oss;
    oss << "{";

    // U_re
    oss << "\"U_re\":[";
    for (int ll_ = 0; ll_ < l; ++ll_) {
        oss << "[";
        for (int ph = 0; ph < 2; ++ph) {
            oss << "[";
            int row = ll_*2 + ph;
            for (int c = 0; c < finalDim; ++c) {
                int base = 2 * (row*finalDim + c);
                double reVal = U[base + 0];
                oss << reVal;
                if (c < finalDim - 1) oss << ",";
            }
            oss << "]";
            if (ph < 1) oss << ",";
        }
        oss << "]";
        if (ll_ < l - 1) oss << ",";
    }
    oss << "],";

    // U_im
    oss << "\"U_im\":[";
    for (int ll_ = 0; ll_ < l; ++ll_) {
        oss << "[";
        for (int ph = 0; ph < 2; ++ph) {
            oss << "[";
            int row = ll_*2 + ph;
            for (int c = 0; c < finalDim; ++c) {
                int base = 2 * (row*finalDim + c);
                double imVal = U[base + 1];
                oss << imVal;
                if (c < finalDim - 1) oss << ",";
            }
            oss << "]";
            if (ph < 1) oss << ",";
        }
        oss << "]";
        if (ll_ < l - 1) oss << ",";
    }
    oss << "],";

    // V -> shape is (n=2*r) x finalDim in row-major, but in Python code
    // we interpret it as (finalDim, 2, r). So dimension order: [d, ph, rr_].
    // d in [0..finalDim-1], ph in [0..1], rr_ in [0..(r-1)].
    // row = rr_*2 + ph, col=d
    // index = row*finalDim + d => base = 2*(row*finalDim + d).

    // Vh_re
    oss << "\"Vh_re\":[";
    for (int d = 0; d < finalDim; ++d) {
        oss << "[";
        for (int ph = 0; ph < 2; ++ph) {
            oss << "[";
            for (int rr_ = 0; rr_ < r; ++rr_) {
                int row = rr_*2 + ph; 
                int base = 2 * (row*finalDim + d);
                double reVal = V[base + 0];
                oss << reVal;
                if (rr_ < r - 1) oss << ",";
            }
            oss << "]";
            if (ph < 1) oss << ",";
        }
        oss << "]";
        if (d < finalDim - 1) oss << ",";
    }
    oss << "],";

    // Vh_im
    oss << "\"Vh_im\":[";
    for (int d = 0; d < finalDim; ++d) {
        oss << "[";
        for (int ph = 0; ph < 2; ++ph) {
            oss << "[";
            for (int rr_ = 0; rr_ < r; ++rr_) {
                int row = rr_*2 + ph;
                int base = 2 * (row*finalDim + d);
                double imVal = V[base + 1];
                oss << imVal;
                if (rr_ < r - 1) oss << ",";
            }
            oss << "]";
            if (ph < 1) oss << ",";
        }
        oss << "]";
        if (d < finalDim - 1) oss << ",";
    }
    oss << "],";

    // Shapes:
    //  "sh":[ [l, finalDim], [finalDim, r] ]
    oss << "\"sh\":[[" << l << "," << finalDim << "],[" 
                       << finalDim << "," << r << "]]";

    oss << "}";

    // Return JSON
    std::string resultStr = oss.str();
    sqlite3_result_text(context, resultStr.c_str(), -1, SQLITE_TRANSIENT);
}

static void svdNoOpFinalize(sqlite3_context *context) {
    // Do nothing—just return NULL to avoid crashes
    sqlite3_result_null(context);
}

/**
 * SQLite extension entry point.
 * Creates the "svd_agg" aggregate with 8 arguments.
 */
extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_svdudf_init(
    sqlite3 *db,
    char **pzErrMsg,
    const sqlite3_api_routines *pApi
){
    SQLITE_EXTENSION_INIT2(pApi);

    // Create the aggregate function: "svd_agg", 8 args
    int rc = sqlite3_create_function(
        db,
        "svd_agg",
        8,
        SQLITE_ANY,
        NULL,
        NULL,         // xFunc
        svdStep,      // xStep
        svdFinalize   // xFinal
    );
    if (rc != SQLITE_OK) {
        if (pzErrMsg) {
            *pzErrMsg = sqlite3_mprintf("Error registering svd_agg: %s", sqlite3_errmsg(db));
        }
        return rc;
    }

    return SQLITE_OK;
    

}

} // extern "C"
