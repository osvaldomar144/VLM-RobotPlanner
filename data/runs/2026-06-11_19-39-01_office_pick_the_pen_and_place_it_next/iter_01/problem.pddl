(define (problem generated_problem)
  (:domain manipulation-base)

  (:objects
    pen - item
  )

  (:init
    (clear pen)
    (gripper-empty)
  )

  (:goal
    (camera-aimed-at pen)
  )
)