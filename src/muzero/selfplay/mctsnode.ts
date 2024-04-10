import {HiddenState} from "../networks/nnet";

/**
 *
 * @hidden
 * @internal
 * @param State An object representing the state of the game.
 */
export class Node<State> {
    // The number of times this node has been visited
    private visits_: number
    // The back propagated value sum of the node
    private valueSum_: number
    // The predicted reward received by moving to this node
    private reward_: number
    // The predicted prior probability of choosing the action that leads to this node
    private prior_: number
    // The hidden state this node corresponds to
    private hiddenState_?: HiddenState
    private readonly possibleActionsLeftToExpand_: Action[]
    private readonly children_: Array<Node<State>> = []
    constructor (
        private readonly state_: State,
        // Possible actions allowed for this state
        possibleActions: Action[],
        // Identification of player to make a move for this state
        private readonly player_: number,
        // Action that caused this state - if node is root no action is defined
        private readonly action_?: Action
    ) {
        this.possibleActionsLeftToExpand_ = possibleActions
        this.reward_ = 0
        this.visits_ = 0
        this.prior_ = 0
        this.valueSum_ = 0
    }

    get possibleActionsLeftToExpand (): Action[] {
        return this.possibleActionsLeftToExpand_
    }

    /**
     * Identification of player to make a move for this state
     */
    get player (): number {
        return this.player_
    }

    /**
     * Action that caused this state - if node is root not action is defined
     */
    get action (): Action | undefined {
        return this.action_
    }

    get children (): Array<Node<State>> {
        return this.children_
    }

    /**
     * The predicted reward received by moving to this node
     */
    get reward (): number {
        return this.reward_
    }

    set reward (value: number) {
        this.reward_ = value
    }

    /**
     * The number of times this node has been visited (updated on each back propagation)
     */
    get visits (): number {
        return this.visits_
    }

    set visits (value: number) {
        this.visits_ = value
    }

    /**
     * The predicted prior probability of choosing the action that leads to this node
     */
    get prior (): number {
        return this.prior_
    }

    set prior (value: number) {
        this.prior_ = value
    }

    /**
     * The back propagated value sum of the node
     */
    get valueSum (): number {
        return this.valueSum_
    }

    set valueSum (value: number) {
        this.valueSum_ = value
    }

    /**
     * The hidden state this node corresponds to
     */
    get hiddenState (): HiddenState {
        if (this.hiddenState_ != null) {
            return this.hiddenState_
        }
        throw new Error('Hidden state is undefined for MCTSState')
    }

    set hiddenState (value: HiddenState) {
        this.hiddenState_ = value
    }

    get state (): State {
        return this.state_
    }

    /**
     * The back propagated value averaged by visit of the node
     */
    value (): number {
        return this.visits_ > 0 ? this.valueSum_ / this.visits_ : 0
    }

    samePlayer (player: number): boolean {
        return this.player_ === player
    }

    policy (actionSpace: number): number[] {
        const totalVisits = this.children.reduce((sum, child) => sum + child.visits, 0)
        const policy: number[] = new Array(actionSpace).fill(0)
        if (totalVisits !== 0) {
            this.children.forEach(child => {
                if (child.action !== undefined) {
                    policy[child.action.id] = child.visits / totalVisits
                }
            })
        }
        return policy
    }

    addChild (
        state: State,
        possibleActions: Action[],
        action: Action,
        // Identification of player to make a move for this state
        player: number
    ): Node<State> {
        const node = new Node(state, possibleActions, player, action)
        this.children_.push(node)
        return node
    }

    isExpanded (): boolean {
        return this.possibleActionsLeftToExpand_.length === 0
    }
}

export interface Action {
    id: number
}
