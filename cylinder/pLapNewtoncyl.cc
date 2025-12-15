/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Copyright (C) 2025 by Hannah Potgieter
 *
 * Author: Hannah Potgieter
 *
 * This code implements a finite element solver for the p-Poisson problem
 * with zero Dirichlet boundary conditions on a cylinder of radius 1 and 
 * height 2 (centered at origin) in 3D. As p approaches infinity, the 
 * solution converges toward the distance-to-boundary function, which 
 * is used for comparison.
 *
 * Large portions of this code are adapted from the deal.II tutorial programs
 * and code gallery, in particular:
 *
 *   - Elastoplastic Torsion (by Salvador Flores)
 *   - step-15
 *   - step-29
 *   
 * deal.II version: 9.6.1
 *
 * deal.II is licensed under the GNU Lesser General Public License
 * (LGPL-2.1-or-later).
 *
 * -----------------------------------------------------------------------------
 */

// Include files
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/geometry_info.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/lac/identity_matrix.h>

#include <typeinfo>
#include <fstream>
#include <iostream>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/base/numbers.h>

// Open a namespace for this program and import everything from the
// dealii namespace into it.
namespace nsp
{
  using namespace dealii;

// ********************************************************//
  class ParameterReader : public Subscriptor
  {
  public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string);
  private:
    void declare_parameters();
    ParameterHandler &prm;
  };
// Constructor
  ParameterReader::ParameterReader(ParameterHandler &paramhandler):
    prm(paramhandler)
  {}

  void ParameterReader::declare_parameters()
  {

    prm.enter_subsection ("Global Parameters");
    {
      prm.declare_entry("p", "100",Patterns::Double(2.1),
                        "Penalization parameter");
      prm.declare_entry("known_solution", "true",Patterns::Bool(),
                        "Whether the exact solution is known");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Mesh & Refinement Parameters");
    {
    prm.declare_entry("Code for the RHS", "0",Patterns::Integer(0,1),
                                 "Number identifying the rhs function");
      prm.declare_entry("No of initial refinements", "4",Patterns::Integer(0),
                        "Number of global mesh refinement steps applied to initial coarse grid");
      prm.declare_entry("No of adaptive refinements", "8",Patterns::Integer(0),
                        "Number of global adaptive mesh refinements");
      prm.declare_entry("top_fraction_of_cells", "0.25",Patterns::Double(0),
                        "refinement threshold");
      prm.declare_entry("bottom_fraction_of_cells", "0.05",Patterns::Double(0),
                        "coarsening threshold");
    }
    prm.leave_subsection ();


    prm.enter_subsection ("Algorithm Parameters");
    {
      prm.declare_entry("init_p", "10",Patterns::Double(2),
                        "Initial p");
      prm.declare_entry("delta_p", "50",Patterns::Double(0),
                        "increase of p");
      prm.declare_entry("Max_CG_it", "1500",Patterns::Integer(1),
                        "Maximum Number of CG iterations");
      prm.declare_entry("CG_tol", "1e-10",Patterns::Double(0),
                        "Tolerance for CG iterations");
      prm.declare_entry("max_LS_it", "45",Patterns::Integer(1),
                        "Maximum Number of LS iterations");
      prm.declare_entry("line_search_tolerence", "1e-6",Patterns::Double(0),
                        "line search tolerance constant (c1 in Nocedal-Wright)");
      prm.declare_entry("init_step_length", "1e-2",Patterns::Double(0),
                        "initial step length in line-search");
      prm.declare_entry("Max_inner", "800",Patterns::Integer(1),
                        "Maximum Number of inner iterations");
      prm.declare_entry("eps", "1.0e-8",Patterns::Double(0),
                        "Threshold on norm of the derivative to declare optimality achieved");
      prm.declare_entry("hi_eps", "1.0e-9",Patterns::Double(0),
                        "Threshold on norm of the derivative to declare optimality achieved in highly refined mesh");
      prm.declare_entry("hi_th", "8",Patterns::Integer(0),
                        "Number of adaptive refinement before change convergence threshold");
    }
    prm.leave_subsection ();

  }
  void ParameterReader::read_parameters (const std::string parameter_file)
  {
    declare_parameters();
    prm.parse_input (parameter_file);
  }

// ******************************************************************************************//

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution (const unsigned int &rhs_value) : Function<dim>(),
    rhs(rhs_value) {}
      
    virtual double value (const Point<dim> &pto, const unsigned int component = 0) const override;
      
    virtual Tensor<1,dim> gradient (const Point<dim> &pto, const unsigned int component = 0) const override;
      
    private:
      const unsigned int rhs;
      
 };


  template <int dim>
  double Solution<dim>::value (const Point<dim> &pto,const unsigned int) const
  {
      
	double r=sqrt(pto[1]*pto[1]+pto[2]*pto[2]);
        
        return 1.0-r;

  }



  template <int dim>
  Tensor<1,dim> Solution<dim>::gradient (const Point<dim> &pto,const unsigned int) const
  {

         double r=sqrt(pto[1]*pto[1]+pto[2]*pto[2]);
          
         Tensor<1,dim> grd;
         grd[0] = 0;
         grd[1] = -1.0*pto[1]/r;
         grd[2] = -1.0*pto[2]/r;
          
         return grd;

  }

// *************************************************************************************** //
  template <int spacedim>
  class pPoisson
  {
  public:
    pPoisson (ParameterHandler &);
    ~pPoisson ();
    void run ();

  private:
    static constexpr unsigned int dim = spacedim;

    void setup_system (const bool initial_step);
    void assemble_system ();
    bool solve (const int inner_it);
    void init_mesh ();
    void refine_mesh ();
    void set_boundary_values ();
    double phi (const double alpha) const;
    bool checkWolfe(double &alpha, double &phi_alpha) const;
    bool determine_step_length (const int inner_it);
    void print_it_message (const int counter, bool ks);
    void output_results (unsigned int refinement) const;
    void format_convergence_tables();
    void process_solution (const unsigned int cycle);

    ParameterHandler &prm;
    Triangulation<dim,  spacedim> triangulation;
    DoFHandler<dim, spacedim> dof_handler;
    
    double L2_error;
    double H1_semi_error;
    double Linfty_error;

    FE_Q<dim, spacedim> fe;
    
    MappingQ<dim, spacedim>      mapping;
 

    AffineConstraints<double> hanging_node_constraints;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    ConvergenceTable convergence_table;
    Vector<double> present_solution;
    Vector<double> newton_update;
    Vector<double> system_rhs;


    double step_length,phi_zero,phi_alpha,phip,phip_zero;
    double old_step,old_phi_zero,old_phip;
    double p;
    double line_search_tolerence; // c_1 in Nocedal & Wright
    std::string elements;
    std::string Method;

  };


template <int dim>
  class Guess : public Function<dim>
  {
  public:
    Guess () : Function<dim>() {}
    
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const override;
  
  };

template <int dim>
  double Guess<dim>::value (const Point<dim> &pto,
                                     const unsigned int /*component*/) const
  {
  
  	(void)pto;
  	
    	return 0.0;
  }

  /*******************************************************************************************/
//                              Boundary condition



  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const override;
  };


  template <int dim>
  double BoundaryValues<dim>::value (
                                     const Point<dim> &pto,
                                     const unsigned int /*component*/) const
  {
          
        (void)pto; 
        
	return 0.0;
      
  }



  /******************************************************************************/
//                       Right-Hand Side
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide (const unsigned int &rhs_value) : Function<dim>() , rhs(rhs_value){}
      
    virtual double value (const Point<dim> &p,
                          const unsigned int component = 0) const override;
  private:
      const unsigned int rhs;
  };

  template <int dim>
  double RightHandSide<dim>::value (const Point<dim> & pto/*p*/,
                                    const unsigned int /*component*/) const
  {
  
	(void)pto; 		
 	 
      if (rhs == 0)
      {
          return 0.0;
      }
      else {
          return 1.0;
      }
  }


  /*******************************************************************/
// The pPoisson class implementation

// Constructor of the class
  template <int spacedim>
  pPoisson<spacedim>::pPoisson (ParameterHandler &param):
    prm(param),
    dof_handler (triangulation),
    L2_error(1.0),
    H1_semi_error(1.0),
    Linfty_error(1.0),
    fe(1),
    mapping(1)
  {
    prm.enter_subsection ("Global Parameters");
    p=prm.get_double("p");
    prm.leave_subsection ();
    prm.enter_subsection ("Algorithm Parameters");
    line_search_tolerence=prm.get_double("line_search_tolerence");
    prm.leave_subsection ();
    if (fe.degree==1)
      elements="P1";
    else elements="P2";
  }



  template <int spacedim>
  pPoisson<spacedim>::~pPoisson ()
  {
    dof_handler.clear ();
  }

  /*****************************************************************************************/
//  print iteration message

  template <int spacedim>
  void pPoisson<spacedim>::print_it_message (const int counter, bool ks)
  {
    if (ks)
      {
        process_solution (counter);
        std::cout << "iteration="<< counter+1 << "  J(u_h)= "<< phi_zero << ", H1 seminorm error: "
                  <<  H1_semi_error  <<", L2_error: "<< L2_error  <<", W0-1,infty error: "<< Linfty_error<< " J'(u_h)(w)= "<< phip
                  << ", |J'(u_h)|= "<< system_rhs.l2_norm()<< ", # dofs: " << dof_handler.n_dofs() <<  ", # cells: " << triangulation.n_active_cells() <<std::endl;
      }
    else
      {
        std::cout << "iteration= " << counter+1 << " J(u_h)= "
                  << phi_alpha << " J'(u_h)= "<< phip<<std::endl;
      }
  }


  /*****************************************************************************************/
//                            Convergence Tables


  /*************************************************************/
// formating

  template <int spacedim>
  void pPoisson<spacedim>::format_convergence_tables()
  {
    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1 semi", 3);
    convergence_table.set_precision("Linfty", 3);
    convergence_table.set_precision("function value", 3);
    convergence_table.set_precision("derivative", 3);

    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1 semi", true);
    convergence_table.set_scientific("Linfty", true);
    convergence_table.set_scientific("function value", true);
    convergence_table.set_scientific("derivative", true);

  }

  /****************************************/
// fill-in entry for the solution
  template <int spacedim>
  void pPoisson<spacedim>::process_solution (const unsigned int it)
  {
    Vector<float> difference_per_cell (triangulation.n_active_cells());

      prm.enter_subsection ("Mesh & Refinement Parameters");
      const int rhs=prm.get_integer("Code for the RHS");
      prm.leave_subsection ();
      
    // compute L2 error (save to difference_per_cell)
    VectorTools::integrate_difference (mapping, dof_handler,present_solution,
                                      Solution<spacedim>(rhs),difference_per_cell,QGauss<dim>(3),VectorTools::L2_norm);
    L2_error = difference_per_cell.l2_norm();

    // compute H1 seminorm error (save to difference_per_cell)
    VectorTools::integrate_difference (mapping, dof_handler,present_solution,Solution<spacedim>(rhs),
                                       difference_per_cell,QGauss<dim>(3),VectorTools::H1_seminorm);
    H1_semi_error = difference_per_cell.l2_norm();

    // compute W1infty error (save to difference_per_cell)
    const QTrapezoid<1> q_trapez;
    const QIterated<dim> q_iterated (q_trapez, 5);
    VectorTools::integrate_difference (mapping, dof_handler,present_solution,Solution<spacedim>(rhs),
                                       difference_per_cell,q_iterated  ,VectorTools::Linfty_norm);
    Linfty_error = difference_per_cell.linfty_norm();
   

    convergence_table.add_value("iteration", it);
    convergence_table.add_value("p", p);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1 semi", H1_semi_error);
    convergence_table.add_value("Linfty", Linfty_error);
    convergence_table.add_value("function value", phi_alpha);
    convergence_table.add_value("derivative", phip);
  }



  /****************************************************************************************/
// pPoisson::setup_system
// unchanged from step-15

  template <int spacedim>
  void pPoisson<spacedim>::setup_system (const bool initial_step)
  {
    if (initial_step)
      {
        dof_handler.distribute_dofs (fe);
        present_solution.reinit (dof_handler.n_dofs());

        VectorTools::interpolate(dof_handler, Guess<spacedim>(), 
                                         present_solution);

        hanging_node_constraints.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler,
                                                 hanging_node_constraints);
        
        hanging_node_constraints.close ();
      }

    // The remaining parts of the function

    newton_update.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    DynamicSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
    hanging_node_constraints.condense (c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);
    system_matrix.reinit (sparsity_pattern);
  }

  /***************************************************************************************/

  template <int spacedim>
  void pPoisson<spacedim>::assemble_system ()
  {

    const QGauss<dim>  quadrature_formula(fe.degree+1);
      
      prm.enter_subsection ("Mesh & Refinement Parameters");
      const int rhs=prm.get_integer("Code for the RHS");;
      prm.leave_subsection ();

    const RightHandSide<spacedim> right_hand_side(rhs);
    
      
    system_matrix = 0;
    system_rhs = 0;

    FEValues<dim, spacedim> fe_values (mapping, fe, quadrature_formula,
                             update_gradients         |
                             update_values           |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    const unsigned int           n_q_points    = quadrature_formula.size();

    FullMatrix<double>           cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>               cell_rhs (dofs_per_cell);

    std::vector<Tensor<1, spacedim> > old_solution_gradients(n_q_points);
    std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);

    hanging_node_constraints.distribute (present_solution);

    typename DoFHandler<dim, spacedim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);
        fe_values.get_function_gradients(present_solution,
                                         old_solution_gradients);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            long double coeff=0.0;
            long double a=old_solution_gradients[q_point] * old_solution_gradients[q_point];
            long double exponent=(p-2.0)/2.0;
            coeff= std::pow(1.0e-5*1.0e-5+a, exponent);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  {
                        cell_matrix(i, j) +=  (fe_values.shape_grad(i, q_point) *  fe_values.shape_grad(j, q_point)
                                              * coeff 
                                              + (p-2.0)
                                              *(fe_values.shape_grad(i, q_point) 
                                              * coeff * std::pow(coeff, -2.0/(p-2.0))
                                              *  (fe_values.shape_grad(j, q_point)
                                              * old_solution_gradients[q_point]) 
                                              * old_solution_gradients[q_point])
                                               )
                                              * fe_values.JxW(q_point);
       
                  }

                cell_rhs(i) -= (  fe_values.shape_grad(i, q_point)
                                  * old_solution_gradients[q_point]
                                  * (coeff) 
                                 -right_hand_side.value(fe_values.quadrature_point(q_point))
                                  *fe_values.shape_value(i, q_point)
                               )
                               * fe_values.JxW(q_point);
              }
          }

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              system_matrix.add (local_dof_indices[i],
                                 local_dof_indices[j],
                                 cell_matrix(i,j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);

          }
        hanging_node_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
        
      }

    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);

    std::map<types::global_dof_index,double> boundary_values;
      
      // boundary
    
      
          VectorTools::interpolate_boundary_values (mapping, dof_handler,
                                                    0,
                                                    dealii::Functions::ZeroFunction<spacedim>(),
                                                    boundary_values);
          
          MatrixTools::apply_boundary_values ( boundary_values,
                                              system_matrix,
                                              newton_update,
                                              system_rhs);
      
  }




  /**********************************      Refine Mesh      ****************************************/
// unchanged from step-15

  template <int spacedim>
  void pPoisson<spacedim>::refine_mesh ()
  {
    using FunctionMap = std::map<types::boundary_id, const Function<spacedim> *>;

    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim, spacedim>::estimate (dof_handler,
                                        QGauss<dim-1>(3),
                                        FunctionMap(),
                                        present_solution,
                                        estimated_error_per_cell);

    prm.enter_subsection ("Mesh & Refinement Parameters");
    const double top_fraction=prm.get_double("top_fraction_of_cells");
    const double bottom_fraction=prm.get_double("bottom_fraction_of_cells");
    prm.leave_subsection ();
    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     top_fraction, bottom_fraction);

    triangulation.prepare_coarsening_and_refinement ();
    SolutionTransfer<dim, Vector<double>, spacedim> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
    triangulation.execute_coarsening_and_refinement();
    dof_handler.distribute_dofs(fe);
    Vector<double> tmp(dof_handler.n_dofs());
    solution_transfer.interpolate(present_solution, tmp);
    present_solution = tmp;
    set_boundary_values ();
    hanging_node_constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();
    hanging_node_constraints.distribute (present_solution);
    setup_system (false);
  }


  /*******************************************************************************************/
// Dump in vtu format for visualization
  template <int spacedim>
  void pPoisson<spacedim>::output_results (unsigned int counter) const
  {
      
      DataOut<dim,  spacedim > data_out;

    const QGauss<dim>  quadrature_formula(3);
        
    Vector<float> difference_per_cell (triangulation.n_active_cells());
      
      prm.enter_subsection ("Mesh & Refinement Parameters");
      const int rhs=prm.get_integer("Code for the RHS");
      prm.leave_subsection ();

    // compute L2 error (save to difference_per_cell)
    VectorTools::integrate_difference (mapping, dof_handler,present_solution,
                                       Solution<spacedim>(rhs),difference_per_cell,QGauss<dim>(3),VectorTools::Linfty_norm);
                                       

    Vector<double> evaluation_point (dof_handler.n_dofs());
    evaluation_point = present_solution;

    FEValues<dim, spacedim> fe_values (fe, quadrature_formula,
                             update_gradients         |
                             update_values    |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    const unsigned int           n_q_points    = quadrature_formula.size();

    std::vector<Tensor<1, spacedim> > gradients(n_q_points);
    std::vector<double> values(n_q_points);

    std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);


    data_out.attach_dof_handler (dof_handler);


    data_out.add_data_vector(
      present_solution,
      "solution",
       DataOut<dim,  spacedim>::type_dof_data);

    data_out.add_data_vector(difference_per_cell,"error");

    data_out.build_patches(mapping, mapping.get_degree());

    std::ostringstream p_str;
    p_str << p<<"-cycle-"<<counter;
    std::string str = p_str.str();
    const std::string filename = "solution-" + str+".vtu";
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);
  }

  /********************************************************************************************/
// unchanged from step-15
  template <int spacedim>
  void pPoisson<spacedim>::set_boundary_values ()
  {
    std::map<types::global_dof_index, double> boundary_values;
      // boundary
     
          VectorTools::interpolate_boundary_values (dof_handler,
                                                 0,
                                                 BoundaryValues<spacedim>(),
                                                 boundary_values);
      
    for (std::map<types::global_dof_index, double>::const_iterator
         bp = boundary_values.begin();
         bp != boundary_values.end(); ++bp)
     present_solution(bp->first) = bp->second;

    hanging_node_constraints.distribute(present_solution);
  }


  /****************************************************************************************/
//  COMPUTE $\phi(\alpha)=J_p(u_h+\alpha w)$
  template <int spacedim>
  double pPoisson<spacedim>::phi (const double alpha) const
  {
    double obj = 0.0;
      
      prm.enter_subsection ("Mesh & Refinement Parameters");
      const int rhs=prm.get_integer("Code for the RHS");;
      prm.leave_subsection ();

    const RightHandSide<spacedim> right_hand_side(rhs);
    
      
    Vector<double> evaluation_point (dof_handler.n_dofs());
    evaluation_point = present_solution;  // copy of u_h
    evaluation_point.add (alpha, newton_update); // u_{n+1}=u_n+alpha w_n

    const QGauss<dim>  quadrature_formula(3);
    FEValues<dim, spacedim> fe_values (fe, quadrature_formula,
                             update_gradients         |
                             update_values    |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    const unsigned int           n_q_points    = quadrature_formula.size();

    Vector<double>               cell_residual (dofs_per_cell);
    std::vector<Tensor<1, spacedim> > gradients(n_q_points);
    std::vector<double> values(n_q_points);


    std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim, spacedim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_residual = 0;
        fe_values.reinit (cell);
        fe_values.get_function_gradients (evaluation_point, gradients);
        fe_values.get_function_values (evaluation_point, values);


        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
            double Du2=gradients[q_point] *  gradients[q_point]; // Du2=|Du|^2
            double penalty;
            if (Du2<1.0e-10)
              penalty=0.0;
            else
              penalty=std::pow(Du2,p/2.0); // penalty=|Du|^p

            // obj+= 1/p |Du|^p -fu
            obj+=(
                   (penalty/p)- right_hand_side.value(fe_values.quadrature_point(q_point))*values[q_point]
                 ) * fe_values.JxW(q_point);
          }

      }

    return obj;
  }


  /*****************************************************************************************/
// check whether putative step-length satisfies sufficient decrease conditions
  template <int spacedim>
  bool pPoisson<spacedim>::checkWolfe(double &alpha, double &phi_alpha) const
  {
    if (phi_alpha< phi_zero+line_search_tolerence*phip*alpha )
      return true;
    else
      return false;
  }


  /*****************************************************************************************/
// Find a step-length satisfying sufficient decrease condition by line-search
// uses quadratic interpolation

  template <int spacedim>
  bool pPoisson<spacedim>::determine_step_length(const int inner_it)
  {
    unsigned int it=0;
    bool done;
    double alpha,nalpha;
    prm.enter_subsection ("Algorithm Parameters");
    const unsigned int max_LS_it=prm.get_integer("max_LS_it");
    double init_SL=prm.get_double("init_step_length");
    prm.leave_subsection ();
    if (inner_it==0)
      alpha=init_SL;
    else
      {
        alpha=std::min(1.45*old_step*old_phip/phip,1.0);
      }
    phi_alpha=phi(alpha);
    std::cerr << "Step length=" << alpha << ", Value= " << phi_alpha;
    // check if step-size satisfies sufficient decrease condition
    done=checkWolfe(alpha,phi_alpha);
    if (done)
      std::cerr << " accepted" << std::endl;
    else
      std::cerr << " rejected" ;

    while ((!done) & (it<max_LS_it))
      {
        // new try obtained by quadratic interpolation
        nalpha=-(phip*alpha*alpha)/(2*(phi_alpha-phi_zero-phip*alpha));

        if (nalpha<1e-3*alpha ||  std::abs(nalpha-alpha)/alpha<1e-8)
          nalpha=alpha/2;
        else if ( phi_alpha-phi_zero>1e3*std::abs(phi_zero) )
          nalpha=alpha/10;
        alpha=nalpha;
        phi_alpha=phi(alpha);
        done=checkWolfe(alpha,phi_alpha);
        if (done)
          std::cerr << ", finished with steplength= "<< alpha<< ", fcn value= "<< phi_alpha<<std::endl;
        it=it+1;
      }
    if (!done)
      {
        std::cerr << ", max. no. of iterations reached with steplength= "<< alpha
                  << ", fcn value= "<< phi_alpha<<std::endl;
        return false;
      }
    else
      {
        step_length=alpha;
        return true;
      }

  }

  /**************************************************************************************************/
  // pPoisson::init_mesh()

  template <int spacedim>
  void pPoisson<spacedim>::init_mesh ()
  {
    // get parameters
    prm.enter_subsection ("Mesh & Refinement Parameters");
    const int init_ref=prm.get_integer("No of initial refinements");
    prm.leave_subsection ();


        // For cylinder of radius 1 and height 2 (centered at origin)
       GridGenerator::cylinder(triangulation, 1.0, 1.0);
       static const CylindricalManifold<dim,spacedim> boundary;
       triangulation.set_manifold (0, boundary);

          
        dof_handler.distribute_dofs(fe);

      
     
     triangulation.refine_global(init_ref);
      
       
  }

  /**************************************************************************************************/
  // pPoisson::solve(inner_it)
  // Performs one inner iteration

  template <int spacedim>
  bool pPoisson<spacedim>::solve (const int inner_it)
  {
    prm.enter_subsection ("Algorithm Parameters");
    const unsigned int max_CG_it=prm.get_integer("Max_CG_it");
    const double CG_tol=prm.get_double("CG_tol");
    prm.leave_subsection ();

    SolverControl solver_control (max_CG_it,CG_tol);
    SolverCG<>    solver (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix,1.2);



    solver.solve (system_matrix, newton_update, system_rhs,
                  preconditioner);
    hanging_node_constraints.distribute (newton_update);
    /******  save current quantities for line-search  **** */
    // Recall that phi(alpha)=J(u+alpha w)
    old_step=step_length;
    old_phi_zero=phi_zero;
    phi_zero=phi(0); // phi(0)=J(u)
    old_phip=phip;
    phip=-1.0*(newton_update*system_rhs); //phi'(0)=J'(u) *w, rhs=-J'(u).
    if (inner_it==0)
      phip_zero=phip;

    if (phip>0)   // this should not happen, step back
      {
        std::cout << "Not a descent direction!" <<std::endl;
        present_solution.add (-1.0*step_length, newton_update);
        step_length=step_length/2;
        phip=old_phip;
        return false;
        //  return true;
      }
    else
      {
        if (determine_step_length(inner_it))
          {
            // update u_{n+1}=u_n+alpha w_n
            present_solution.add (step_length, newton_update);
            return true;
          }
        else return false;
      }
  }




  /*************************************************************************************************************/
// pPoisson::run
  template <int spacedim>
  void pPoisson<spacedim>::run ()
  {

    // get parameters
    prm.enter_subsection ("Mesh & Refinement Parameters");
    const int adapt_ref=prm.get_integer("No of adaptive refinements");
    prm.leave_subsection ();
    prm.enter_subsection ("Algorithm Parameters");
    const int max_inner=prm.get_integer("Max_inner");
    const double eps=prm.get_double("eps");
    const double hi_eps=prm.get_double("hi_eps");
    const int hi_th=prm.get_integer("hi_th");
    const double init_p=prm.get_double("init_p");
    const double delta_p=prm.get_double("delta_p");
    prm.leave_subsection ();
    prm.enter_subsection ("Global Parameters");
    bool known_solution=prm.get_bool("known_solution");
    double actual_p=prm.get_double("p");
    prm.leave_subsection ();
    /************************/

    // init Timer
    Timer timer;
    double ptime=0.0;
    double totaltime =0.0;
    timer.start ();

    // initalize mesh for the selected domain
    init_mesh();

    // setup FE space
    setup_system (true);
    set_boundary_values ();

    // init counters
    int global_it=0;    // Total inner iterations (counting both loops)
    int cycle=0;          // Total outer iterations (counting both loops)
    int refinement = 0;    //  Refinements  performed (adaptive) = outer iterations 2nd loop


    // prepare to start first loop
    p=init_p;
    bool well_solved=true;

    /*****************************          First loop      ***********************************/
    /****************** Prepare initial condition using increasing p  *************************/
    while (p<=actual_p) // outer iteration, increasing p.
      {
        std::cout <<"--Preparing initial condition with p="<<p<<" iter.= " << global_it<<  "  .-- "<< std::endl;
        timer.restart();
        for (int inner_iteration=0; inner_iteration<max_inner; ++inner_iteration,++global_it)
          {
            assemble_system ();
            std::cout << "constraints:" << hanging_node_constraints.n_constraints() << std::endl;
            well_solved=solve (inner_iteration);
            print_it_message (global_it, known_solution);
            if (
              ((system_rhs.l2_norm()/std::sqrt(system_rhs.size()) <1e-9) & (cycle<1)) |
              ((system_rhs.l2_norm()/std::sqrt(system_rhs.size()) <1e-7) & (cycle>=1)) |
              !well_solved
            )
              break;

              
          }
        ptime=timer.cpu_time();
          totaltime = totaltime+ptime;
        std::cout << "TIME ON CURRENT p VALUE: " << ptime << std::endl;
        //if (well_solved)
          output_results(cycle);
        //refine_mesh();      
        cycle++;
        p+=delta_p;
      }
    /***************************    first loop finished        ********************/


    // prepare for second loop
    p=actual_p;
    well_solved=true;


    /*****************************        Second loop         *********************************/
    /**************************** Solve problem for target p  *********************************/

    std::cout << "============ Finish problem with p="   <<p << " =================="  << std::endl;
    std::cout << "constraints:" << hanging_node_constraints.n_constraints() << std::endl;
    /*****    Outer iteration - refining mesh  ******************/
    while ((cycle<adapt_ref) & well_solved)
      {
        timer.restart();
        // inner iteration
        for (int inner_iteration=0; inner_iteration<max_inner; ++inner_iteration,++global_it)
          {
            assemble_system ();
            well_solved=solve (inner_iteration);
            print_it_message (global_it, known_solution);

            if (
              ((system_rhs.l2_norm()/std::sqrt(system_rhs.size()) < eps) & (refinement<hi_th)) |
              (( system_rhs.l2_norm()/ std::sqrt (system_rhs.size()) <hi_eps)  | (!well_solved))
            )
              break;
          }
        //inner iterations finished
        ptime=timer.cpu_time();
          totaltime = totaltime+ptime;
        std::cout << "TIME ON CURRENT p VALUE: " << ptime << std::endl;
        //if (well_solved)
          output_results (cycle);
                  // update counters
        ++refinement;
        ++cycle;
        // refine mesh
        std::cout << "******** Refined mesh " << cycle    << " ********"  << std::endl;
        refine_mesh();
      }// second loop

      std::cout << "TOTAL TIME ELAPSED: " << totaltime << std::endl;
      
    // write convergence tables to file
    if (known_solution)
      {
        format_convergence_tables();
        std::string error_filename = "error"+Method+elements+".tex";
        std::ofstream error_table_file(error_filename.c_str());
        convergence_table.write_tex(error_table_file);
      }
  }//run()

}//namespace

/**********************************************************************************************/
// The main function
int main ()
{
  try
    {
      using namespace dealii;
      using namespace nsp;
      deallog.depth_console (0);

      ParameterHandler prm;
      ParameterReader param(prm);
      param.read_parameters("PARAMScyl.prm");
        prm.enter_subsection ("Mesh & Refinement Parameters");
        prm.leave_subsection ();
            std::cout << "Instantiating pPoisson<3>" << std::endl;
            pPoisson<3> pPoissonProblem(prm);
            pPoissonProblem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"    << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
