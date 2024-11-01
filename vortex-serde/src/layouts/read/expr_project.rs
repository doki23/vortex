use std::sync::Arc;

use vortex_dtype::field::Field;
use vortex_expr::{BinaryExpr, Column, Identity, Literal, Operator, Select, VortexExpr};

use crate::layouts::RowFilter;

/// Restrict expression to only the fields that appear in projection
pub fn expr_project(
    expr: &Arc<dyn VortexExpr>,
    projection: &[Field],
) -> Option<Arc<dyn VortexExpr>> {
    if let Some(rf) = expr.as_any().downcast_ref::<RowFilter>() {
        rf.only_fields(projection).map(|rf| Arc::new(rf) as _)
    } else if expr.as_any().downcast_ref::<Literal>().is_some() {
        Some(expr.clone())
    } else if let Some(s) = expr.as_any().downcast_ref::<Select>() {
        match s {
            Select::Include(i) => {
                let fields = i
                    .iter()
                    .filter(|f| projection.contains(f))
                    .cloned()
                    .collect::<Vec<_>>();
                if projection.len() == 1 {
                    Some(Arc::new(Identity))
                } else {
                    (!fields.is_empty()).then(|| Arc::new(Select::include(fields)) as _)
                }
            }
            Select::Exclude(e) => {
                let fields = projection
                    .iter()
                    .filter(|f| !e.contains(f))
                    .cloned()
                    .collect::<Vec<_>>();
                if projection.len() == 1 {
                    Some(Arc::new(Identity))
                } else {
                    (!fields.is_empty()).then(|| Arc::new(Select::include(fields)) as _)
                }
            }
        }
    } else if let Some(c) = expr.as_any().downcast_ref::<Column>() {
        projection.contains(c.field()).then(|| {
            if projection.len() == 1 {
                Arc::new(Identity)
            } else {
                expr.clone()
            }
        })
    } else if let Some(bexp) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let lhs_proj = expr_project(bexp.lhs(), projection);
        let rhs_proj = expr_project(bexp.rhs(), projection);
        if bexp.op() == Operator::And {
            match (lhs_proj, rhs_proj) {
                (Some(lhsp), Some(rhsp)) => Some(Arc::new(BinaryExpr::new(lhsp, bexp.op(), rhsp))),
                // Projected lhs and rhs might lose reference to columns if they're simplified to straight column comparisons
                (Some(lhsp), None) => (!bexp
                    .rhs()
                    .references()
                    .intersection(&bexp.lhs().references())
                    .any(|f| projection.contains(f)))
                .then_some(lhsp),
                (None, Some(rhsp)) => (!bexp
                    .lhs()
                    .references()
                    .intersection(&bexp.rhs().references())
                    .any(|f| projection.contains(f)))
                .then_some(rhsp),
                (None, None) => None,
            }
        } else {
            Some(Arc::new(BinaryExpr::new(lhs_proj?, bexp.op(), rhs_proj?)))
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::field::Field;
    use vortex_expr::{BinaryExpr, Column, Identity, Literal, Operator, Select, VortexExpr};

    use crate::layouts::read::expr_project::expr_project;

    #[test]
    fn project_and() {
        let band = Arc::new(BinaryExpr::new(
            Arc::new(Column::new(Field::from("a"))),
            Operator::And,
            Arc::new(Column::new(Field::from("b"))),
        )) as _;
        let projection = vec![Field::from("b")];
        assert_eq!(
            *expr_project(&band, &projection).unwrap(),
            *Identity.as_any()
        );
    }

    #[test]
    fn project_or() {
        let bor = Arc::new(BinaryExpr::new(
            Arc::new(Column::new(Field::from("a"))),
            Operator::Or,
            Arc::new(Column::new(Field::from("b"))),
        )) as _;
        let projection = vec![Field::from("b")];
        assert!(expr_project(&bor, &projection).is_none());
    }

    #[test]
    fn project_nested() {
        let band = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new(Field::from("a"))),
                Operator::Lt,
                Arc::new(Column::new(Field::from("b"))),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(Literal::new(5.into())),
                Operator::Lt,
                Arc::new(Column::new(Field::from("b"))),
            )),
        )) as _;
        let projection = vec![Field::from("b")];
        assert!(expr_project(&band, &projection).is_none());
    }

    #[test]
    fn project_multicolumn() {
        let blt = Arc::new(BinaryExpr::new(
            Arc::new(Column::new(Field::from("a"))),
            Operator::Lt,
            Arc::new(Column::new(Field::from("b"))),
        )) as _;
        let projection = vec![Field::from("a"), Field::from("b")];
        assert_eq!(
            *expr_project(&blt, &projection).unwrap(),
            *BinaryExpr::new(
                Arc::new(Column::new(Field::from("a"))),
                Operator::Lt,
                Arc::new(Column::new(Field::from("b"))),
            )
            .as_any()
        );
    }

    #[test]
    fn project_select() {
        let blt = Arc::new(Select::include(vec![
            Field::from("a"),
            Field::from("b"),
            Field::from("c"),
        ])) as _;
        let projection = vec![Field::from("a"), Field::from("b")];
        assert_eq!(
            *expr_project(&blt, &projection).unwrap(),
            *Select::include(projection).as_any()
        );
    }

    #[test]
    fn project_select_extra_columns() {
        let blt = Arc::new(Select::include(vec![
            Field::from("a"),
            Field::from("b"),
            Field::from("c"),
        ])) as _;
        let projection = vec![Field::from("c"), Field::from("d")];
        assert_eq!(
            *expr_project(&blt, &projection).unwrap(),
            *Select::include(vec![Field::from("c")]).as_any()
        );
    }
}
