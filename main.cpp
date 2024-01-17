#include <Kokkos_Core.hpp>
#include <mpicpp.hpp>
#include "gtest/gtest.h"
#include <p3a_static_matrix.hpp>


#define SIZE 512
//#define SIZE 16
#define BLOCK_SIZE 8


#define T double
#define MY_FLOAT_TYPE double
#define MY_BOOLEAN_TYPE _Bool
#define MY_SIZE_TYPE size_t

namespace Benchmark
{
    char str[128];

    auto results_before = std::make_tuple(-1.0e+0, 1.0e+0, 1.0e+0);
    auto results_after = std::make_tuple(-1.0e+0, 1.0e+0, 1.0e+0);

    void generate_matrix_data(p3a::static_matrix<T, SIZE, SIZE> * my_matrix)
    {
        for (MY_SIZE_TYPE i = 0; i < SIZE; i++)
        {
            for (MY_SIZE_TYPE j = 0; j < SIZE; j++)
            {
    	           (*my_matrix)(i, j) = i * j;
            }
        }
    }


    void generate_matrix_zeros(p3a::static_matrix<T, SIZE, SIZE> * my_matrix)
    {
        for (MY_SIZE_TYPE i = 0; i < SIZE; i++)
        {
            for (MY_SIZE_TYPE j = 0; j < SIZE; j++)
            {
                (*my_matrix)(i, j) = 0;
            }
        }
    }


    void base_algorithm_before_optimization(p3a::static_matrix<T, SIZE, SIZE> * x, p3a::static_matrix<T, SIZE, SIZE> * y, p3a::static_matrix<T, SIZE, SIZE> * z)
    {
      T result = 0;
      for (MY_SIZE_TYPE i = 0; i < SIZE; i++)
      {
          for (MY_SIZE_TYPE j = 0; j < SIZE; j++)
          {
              result = 0;
              for (MY_SIZE_TYPE k = 0; k < SIZE; k++)
              {
                  result = result + (*y)(i, k) * (*z)(k, j);
              }
              (*x)(i, j) = result;
          }
      }
    }




    void base_algorithm_after_optimization(p3a::static_matrix<T, SIZE, SIZE> * x, p3a::static_matrix<T, SIZE, SIZE> * y, p3a::static_matrix<T, SIZE, SIZE> * z)
    {
        T result = 0;
        for (MY_SIZE_TYPE jj = 0; jj < SIZE; jj = jj + BLOCK_SIZE)
        {
            for (MY_SIZE_TYPE kk = 0; kk < SIZE; kk = kk + BLOCK_SIZE)
            {
                for (MY_SIZE_TYPE i = 0; i < SIZE; i++)
                {
                    for (MY_SIZE_TYPE j = jj; j < (jj + BLOCK_SIZE < SIZE ? jj + BLOCK_SIZE : SIZE); j++)
                    {
                        result = 0;
                        for (MY_SIZE_TYPE k = kk; k < (kk + BLOCK_SIZE < SIZE ? kk + BLOCK_SIZE : SIZE); k++)
                        {
    		                    result = result + (*y)(i, k) * (*z)(k, j);
                        }
                        (*x)(i, j) = (*x)(i, j) + result;
                    }
                }
            }
        }
    }

    std::tuple<T, MY_FLOAT_TYPE, MY_FLOAT_TYPE> initialize_problem_and_run(p3a::static_matrix<T, SIZE, SIZE> * x, p3a::static_matrix<T, SIZE, SIZE> * y, p3a::static_matrix<T, SIZE, SIZE> * z, const MY_BOOLEAN_TYPE & using_optimization)
    {
        generate_matrix_zeros(x);
        fflush(stdout);
        generate_matrix_data(y);
        generate_matrix_data(z);

        const MY_FLOAT_TYPE N = SIZE;

        auto blocking_start_time = clock();
        auto blocking_end_time = clock();

        if(using_optimization)
        {
            blocking_start_time = clock();
            base_algorithm_after_optimization(x, y, z);
            blocking_end_time = clock();
        }

        if(!using_optimization)
        {
            blocking_start_time = clock();
            base_algorithm_before_optimization(x, y, z);
            blocking_end_time = clock();
        }

        const MY_FLOAT_TYPE blocking_duration = (MY_FLOAT_TYPE)(blocking_end_time - blocking_start_time) / CLOCKS_PER_SEC;

        T max_element = -1;
        for (MY_SIZE_TYPE i = 0; i < SIZE; i++)
        {
            for (MY_SIZE_TYPE j = 0; j < SIZE; j++)
            {
                max_element = max_element < ((*x)(i, j)) ? ((*x)(i, j)) : max_element;
            }
        }

      return std::make_tuple(max_element, blocking_duration, (2*N*N*N/blocking_duration));
    }


    std::tuple<T, MY_FLOAT_TYPE, MY_FLOAT_TYPE> run_algorithm_and_get_results(const MY_BOOLEAN_TYPE & using_optimization)
    {
        p3a::static_matrix<T, SIZE, SIZE> * x = new p3a::static_matrix<T, SIZE, SIZE>;
        p3a::static_matrix<T, SIZE, SIZE> * y = new p3a::static_matrix<T, SIZE, SIZE>;
        p3a::static_matrix<T, SIZE, SIZE> * z = new p3a::static_matrix<T, SIZE, SIZE>;

        const auto my_results = initialize_problem_and_run(x, y, z, using_optimization);

        delete x;
        delete y;
        delete z;

        return my_results;
    }



    class [[nodiscard]] environment
    {
    public:
        environment(int* argc, char*** argv);
        environment() : environment(nullptr, nullptr) {}
        ~environment();

        environment(environment const&) = delete;
        environment& operator=(environment const&) = delete;
        environment(environment&&) = delete;
        environment& operator=(environment&&) = delete;

    private:
        void cleanupResources();
    };

    class MyBenchmarkEnvironment : public ::testing::Environment
    {
    public:
        virtual void TearDown() override{}
    };

    environment::environment(int* argc, char*** argv)
    {
        if(argc && argv)
        {
            results_before = run_algorithm_and_get_results(0);
            results_after = run_algorithm_and_get_results(1);
        }
    }

    class MyBenchmarkSuite : public ::testing::Test
    {
    protected:
        static void SetUpTestSuite() {}

        static void TearDownTestSuite() {}
    };




    MY_BOOLEAN_TYPE check_if_max_elements_are_the_same(const T & second_max_element, const T & first_max_element)
    {
        return ((second_max_element) == (first_max_element));
    }


    MY_BOOLEAN_TYPE check_if_before_has_longer_duration(const MY_FLOAT_TYPE & run_time_duration_after, const MY_FLOAT_TYPE & run_time_duration_before)
    {
        return ((run_time_duration_before) > (run_time_duration_after));
    }


    MY_BOOLEAN_TYPE check_if_before_has_less_flops(const MY_FLOAT_TYPE & flops_after, const MY_FLOAT_TYPE & flops_before)
    {
        return ((flops_before) < (flops_after));
    }


    MY_BOOLEAN_TYPE check_if_after_has_shorter_duration(const MY_FLOAT_TYPE & run_time_duration_after, const MY_FLOAT_TYPE & run_time_duration_before)
    {
        return ((run_time_duration_after) < (run_time_duration_before));
    }


    MY_BOOLEAN_TYPE check_if_after_has_more_flops(const MY_FLOAT_TYPE & flops_after, const MY_FLOAT_TYPE & flops_before)
    {
        return ((flops_after) > (flops_before));
    }


    MY_FLOAT_TYPE calculate_ratio_duration_after_duration_before(const MY_FLOAT_TYPE & run_time_duration_after, const MY_FLOAT_TYPE & run_time_duration_before)
    {
        return ((run_time_duration_after) / (run_time_duration_before));
    }


    MY_FLOAT_TYPE calculate_ratio_flops_after_flops_before(const MY_FLOAT_TYPE & flops_after, const MY_FLOAT_TYPE & flops_before)
    {
        return ((flops_after) / (flops_before));
    }


    MY_FLOAT_TYPE calculate_euclidian_distance(const MY_FLOAT_TYPE & xa, const MY_FLOAT_TYPE & xb, const MY_FLOAT_TYPE & ya, const MY_FLOAT_TYPE & yb)
    {
        return sqrt(((xb-xa)*(xb-xa)) + ((yb-ya)*(yb-ya)));
    }

    MY_FLOAT_TYPE calculate_geometric_mean(const MY_FLOAT_TYPE & a, const MY_FLOAT_TYPE & b)
    {
        return sqrt(a*b);
    }

    TEST_F(MyBenchmarkSuite, MaxElementsAreExactlyTheSame)
    {
        ASSERT_TRUE(check_if_max_elements_are_the_same(std::get<0>(results_after), std::get<0>(results_before)));
    }

    TEST_F(MyBenchmarkSuite, MaxElementsAreTheSame)
    {
        EXPECT_DOUBLE_EQ(std::get<0>(results_after), std::get<0>(results_before));
    }


    TEST_F(MyBenchmarkSuite, BeforeHasLongerDuration)
    {
        EXPECT_TRUE(check_if_before_has_longer_duration(std::get<1>(results_after), std::get<1>(results_before)));
    }

    TEST_F(MyBenchmarkSuite, BeforeHasLessFlops)
    {
        EXPECT_TRUE(check_if_before_has_less_flops(std::get<2>(results_after), std::get<2>(results_before)));
    }

    TEST_F(MyBenchmarkSuite, AfterHasShorterDuration)
    {
        EXPECT_TRUE(check_if_after_has_shorter_duration(std::get<1>(results_after), std::get<1>(results_before)));
    }

    TEST_F(MyBenchmarkSuite, AfterHasMoreFlops)
    {
        EXPECT_TRUE(check_if_after_has_more_flops(std::get<2>(results_after), std::get<2>(results_before)));
    }

    TEST_F(MyBenchmarkSuite, RatioDurationAfterDurationBeforeIsLessThanOne)
    {
        EXPECT_LT(calculate_ratio_duration_after_duration_before(std::get<1>(results_after), std::get<1>(results_before)), 1.0e+0);
    }


    TEST_F(MyBenchmarkSuite, RatioFLOPSAfterFLOPSBeforeIsGreaterThanOne)
    {
        EXPECT_GT(calculate_ratio_flops_after_flops_before(std::get<2>(results_after), std::get<2>(results_before)), 1.0e+0);
    }

    environment::~environment()
    {
        cleanupResources();
    }

    void environment::cleanupResources()
    {
        const MY_FLOAT_TYPE geometric_mean_average_duration = calculate_geometric_mean(std::get<1>(results_after), std::get<1>(results_before));
        const MY_FLOAT_TYPE geometric_mean_average_flops = calculate_geometric_mean(std::get<2>(results_after), std::get<2>(results_before));


        sprintf(str, "%e", geometric_mean_average_duration);
        printf("\ngeometric_mean_average_duration:                  %s (seconds avg)\n\r", str);

        sprintf(str, "%e", geometric_mean_average_flops);
        printf("geometric_mean_average_flops:                     %s (flops avg)\n\n\r", str);
    }

}



int main(int argc, char * argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  mpicpp::environment mpi_state(&argc, &argv);
  Benchmark::environment Benchmark_state(&argc, &argv);
  Kokkos::ScopeGuard kokkos_library_state(argc, argv);
  return RUN_ALL_TESTS();
}
